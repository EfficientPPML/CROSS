import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as m
from native_capture import NativeCapture

torch.set_num_threads(1)

def tiny():
    torch.manual_seed(291)
    config=Qwen3Config(vocab_size=37,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,
        attention_dropout=0.,attention_bias=False,hidden_act='silu',tie_word_embeddings=True)
    config._attn_implementation='eager'
    return Qwen3ForCausalLM(config).float().eval()

def inputs(sequence=4):
    ids=torch.arange(2*sequence).reshape(2,sequence)%37
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(sequence)[None,:].expand(2,-1),use_cache=False)

def module_hooks(model):
    return {id(module):tuple(module._forward_hooks) for module in model.modules()}

class NativeTests(unittest.TestCase):
    def test_complete_native_outputs_and_unmodified_logits(self):
        model=tiny();arguments=inputs();original=m.eager_attention_forward;hooks=module_hooks(model)
        with torch.no_grad():baseline=model(**arguments).logits.clone()
        with NativeCapture(model) as capture,torch.no_grad():
            capture.begin();actual=model(**arguments).logits;blocks,ops=capture.end()
            np.testing.assert_array_equal(actual.numpy(),baseline.numpy())
            self.assertEqual(len(ops),31);self.assertEqual(len(blocks),2)
            self.assertEqual(ops['0.q.inverse_sqrt'].shape,(2,4,4,8))
            self.assertEqual(ops['0.k.inverse_sqrt'].shape,(2,2,4,8))
            self.assertEqual(ops['0.gqa'].shape,(2,4,32))
            self.assertEqual(ops['final.inverse_sqrt'].shape,(2,16))
            # Independently execute one actual native block, with no active
            # capture, and compare each exposed native operator's tensor.
            layer=model.model.layers[0];a=layer.self_attn;x=model.model.embed_tokens(arguments['input_ids'])
            z=layer.input_layernorm(x);qraw=a.q_proj(z);kraw=a.k_proj(z);vraw=a.v_proj(z)
            q=a.q_norm(qraw.reshape(2,4,4,8)).transpose(1,2)
            k=a.k_norm(kraw.reshape(2,4,2,8)).transpose(1,2)
            v=vraw.reshape(2,4,2,8).transpose(1,2)
            cos,sin=model.model.rotary_emb(x,arguments['position_ids']);qr,kr=m.apply_rotary_pos_emb(q,k,cos,sin)
            scores=torch.matmul(qr,m.repeat_kv(kr,2).transpose(2,3))*a.scaling
            mask=torch.triu(torch.full((4,4),torch.finfo(torch.float32).min),diagonal=1)
            probabilities=torch.softmax(scores+mask,dim=-1,dtype=torch.float32)
            attended=(probabilities@m.repeat_kv(v,2)).transpose(1,2).contiguous().reshape(2,4,32)
            branch=a.o_proj(attended);residual=x+branch;post=layer.post_attention_layernorm(residual)
            gate=layer.mlp.gate_proj(post);up=layer.mlp.up_proj(post);silu=layer.mlp.act_fn(gate);down=layer.mlp.down_proj(silu*up)
            independent={'input.inverse_sqrt':z,'q_linear':qraw,'k_linear':kraw,'v_linear':vraw,
                'q.inverse_sqrt':q,'k.inverse_sqrt':k,'q_rope':qr,'k_rope':kr,'gqa':attended,
                'output_linear':branch,'post.inverse_sqrt':post,'gate_linear':gate,'up_linear':up,'silu':silu,'gated_mlp':down}
            for name,tensor in independent.items():
                with self.subTest(operator=name):np.testing.assert_array_equal(ops['0.'+name],tensor.numpy().astype(np.float64))
            np.testing.assert_array_equal(blocks[0],(residual+down).numpy().astype(np.float64))
            np.testing.assert_array_equal(ops['final.inverse_sqrt'],model.model.norm(torch.from_numpy(blocks[-1]).float())[:,-1].numpy().astype(np.float64))
            for value in (*blocks,*ops.values()):self.assertEqual(value.dtype,np.float64)
            ops['0.q_linear'].fill(123.)
            np.testing.assert_array_equal(actual.numpy(),baseline.numpy())
            capture.begin();model(**inputs(1));blocks2,ops2=capture.end();self.assertEqual(ops2['0.gqa'].shape,(2,1,32))
        self.assertIs(m.eager_attention_forward,original);self.assertEqual(module_hooks(model),hooks)

    def test_cancel_and_partial_registration_restore_everything(self):
        model=tiny();original=m.eager_attention_forward;hooks=module_hooks(model)
        cancellation=KeyboardInterrupt('primary')
        calls=[]
        class Broken:
            def remove(self):calls.append('broken');raise RuntimeError('cleanup')
        class Later:
            def remove(self):calls.append('later')
        with self.assertRaises(KeyboardInterrupt) as caught:
            with NativeCapture(model) as capture:
                capture.handles.extend([Broken(),Later()]);capture.begin();raise cancellation
        self.assertIs(caught.exception,cancellation);self.assertEqual(calls,['broken','later'])
        self.assertTrue(any('cleanup' in note for note in cancellation.__notes__))
        self.assertIs(m.eager_attention_forward,original);self.assertEqual(module_hooks(model),hooks)
        with patch.object(model.model.layers[0].self_attn.q_norm,'register_forward_hook',side_effect=RuntimeError('register')):
            with self.assertRaisesRegex(RuntimeError,'register'):NativeCapture(model).__enter__()
        self.assertIs(m.eager_attention_forward,original);self.assertEqual(module_hooks(model),hooks)

    def test_incomplete_duplicate_nested_and_invalid_phase_rejected(self):
        model=tiny();capture=NativeCapture(model)
        with self.assertRaises(ValueError):capture.begin()
        with capture:
            with self.assertRaisesRegex(ValueError,'nested'):NativeCapture(model).__enter__()
            capture.begin()
            with self.assertRaisesRegex(ValueError,'inactive'):capture.begin()
            with self.assertRaisesRegex(ValueError,'incomplete'):capture.end()
            capture.begin();capture._store('0.q_linear',torch.zeros(1))
            with self.assertRaisesRegex(ValueError,'duplicate'):capture._store('0.q_linear',torch.zeros(1))
        self.assertFalse(capture.active)

    def test_attention_contract_masks_and_cached_shapes(self):
        model=tiny();attention=model.model.layers[0].self_attn;q=torch.ones((2,4,2,8));k=torch.ones((2,2,2,8));v=k.clone()
        with NativeCapture(model) as capture:
            for mask in (None,torch.zeros(2,1,2,2),torch.tensor([[0.,1.],[0.,0.]]),torch.tensor([[0.,-.1],[0.,0.]])):
                capture.begin()
                with self.subTest(mask=str(mask)),self.assertRaises((ValueError,IndexError)):
                    m.eager_attention_forward(attention,q,k,v,mask,attention.scaling)
                capture.active=False
            capture.begin()
            with self.assertRaisesRegex(ValueError,'KV cache'):m.eager_attention_forward(attention,q,k[:,:,:1],v,None,attention.scaling)
        with self.assertRaisesRegex(ValueError,'float32'):NativeCapture._copy(torch.ones(1,dtype=torch.float64))
        with self.assertRaisesRegex(ValueError,'nonfinite'):NativeCapture._copy(torch.tensor([float('nan')]))

if __name__=='__main__':unittest.main()

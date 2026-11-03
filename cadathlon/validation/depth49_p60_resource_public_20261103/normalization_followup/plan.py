"""Read-only public normalization/error accounting; no CKKS context, keys or HE."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import hashlib,io,json,math,os,tarfile
HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('CROSS_ROOT','/home/jianming_gatech/CROSS'))
PROFILE=REPO/'cadathlon/validation/depth49_p60_resource_public_20261103/report.json'
ADAPTER=REPO/'cadathlon/common/encrypted_refresh.py'
BOOT=REPO/'jaxite_word/bootstrapping.py'

def sha(data):return hashlib.sha256(data).hexdigest()
def up(x):
 v=float(x)
 return math.nextafter(v,math.inf) if F(v)<x else v

def main():
 paths=[Path(__file__),PROFILE,ADAPTER,BOOT];before={str(p):sha(p.read_bytes()) for p in paths}
 profiles=json.loads(PROFILE.read_text())['profiles'];rows=[]
 for name in ('depth49_p56_d4','depth49_p60_d6','depth49_p60_d8'):
  p=next(p for p in profiles if p['name']==name);q=p['q'];delta=p['canonical_scale'];Q=math.prod(q[:2]);sf0=delta
  # Exact transcription of Bootstrap.control_gen public scalar selection.
  powP=2.**round(math.log2(sf0));degree=round(math.log2(float(Q)/powP)) if float(Q)>powP else 0
  correction_factor=max(7,min(14,round(-.2419*(2*math.log2(p['degree'])+math.log2(p['slots']))+19.081)))
  correction=correction_factor-degree
  capacity=F(Q)/(2*F(sf0));drop=q[3]*float(q[2])
  # Actual public binary64 scalar expression from _adjust_ciphertext. The scalar
  # encoder's coefficient rounding and ciphertext/rescale noise remain separate.
  source_scales=[math.nextafter(delta,0.),delta,math.nextafter(delta,math.inf)]
  gains=[]
  for source in source_scales:
   adjustment=(sf0/source)*(drop/source)*math.pow(2,-correction)
   gain=F(source)*F(source)*F(adjustment)/(q[2]*q[3]*F(sf0))
   gains.append(gain)
  gain=max(gains)
  candidates=[]
  for bound in (16,32):
   amplitude=F(16,bound);base=amplitude*gain
   candidates.append(dict(public_original_amplitude=16,normalization_factor=bound,
    normalized_amplitude=up(amplitude),ideal_adjust_gain=2.**(-correction),rounded_public_adjust_gain_upper=up(gain),
    adjusted_base_message_amplitude_upper=up(base),base_signed_capacity=up(capacity),
    public_error_headroom_before_wrap=float(capacity-base),satisfies_base_capacity_before_unbounded_noise=base<capacity,
    input_preconditioning_levels=1,normalized_output_level=25,restored_absolute_error_multiplier=bound,
    comment='Headroom is public message-space subtraction, not a certified bound on encryption/bootstrap error.'))
  assert candidates[0]['satisfies_base_capacity_before_unbounded_noise']
  rows.append(dict(name=name,base_Q=Q,canonical_scale=delta,unadjusted_base_Q_over_scale=float(F(Q)/F(delta)),
   unadjusted_signed_capacity=float(capacity),source_scale_probes=source_scales,fullslot_correctionFactor=correction_factor,degree_correction=degree,
   adjust_correction=correction,post_S2C_integer_restoration=2**correction,
   first_meta_pass_error_capacity_condition=dict(meta_precision=10,
      normalized_first_pass_coefficient_error_sufficient_upper=float(capacity/(gain*1024)),
      condition='Conservative sufficient coefficient-envelope no-wrap condition for the second depleted error input after adjustment, ignoring further alignment/ciphertext error. Not a necessary condition for every slot pattern or an accuracy certificate.'),
   candidates=candidates))
 report=dict(scope=__doc__,passed=True,created_utc=datetime.now(timezone.utc).isoformat(),profiles=rows,
  conclusion='The bare baseQ/scale comparison is insufficient. Existing actual AdjustCiphertext attenuates before base reduction. For the stated full-slot profiles, B=16 already fits the base signed capacity; this argument does not require B=32.',
  helper_behavior=['BoundedSecureRefresh checks two remaining input levels; when B!=1 it performs an actual weighted_sum by1/B, then lowers to level1.',
   'Bootstrap._adjust_ciphertext always runs on this level1/CD2 input, applies the actual public adjustment scalar and rescale, then sets the documented sf0 metadata. Its ideal message effect is2^(-correction).',
   'The bootstrap post-S2C integer multiply restores2^correction with no extra level. Meta10 performs a second bootstrap on amplified first-pass error.',
   'Full-slot period65536 has no periodic averaging. Final weighted_sum byB restores the public range and canonical scale in one actual level.',
   'B16 andB32 therefore have identical input/output level contracts here; B32 doubles worst-case absolute error amplification from a fixed normalized-error bound.'],
  error_accounting=dict(before_meta='Relative to an original plaintextx with input error epsilon_in: preconditioned message=(x+epsilon_in)/B+epsilon_pre.',
   final='With normalized bootstrap error epsilon_boot and final alignment error epsilon_post: output=x+epsilon_in+B*(epsilon_pre+epsilon_boot)+epsilon_post.',
   meta_ideal='Ideal two-pass cancellation gives normalized epsilon_boot=-epsilon_second/1024; actual alignment/arithmetic errors add terms and need measurement.',
   comparison='Choosing largerB may change internal approximation errors, but cannot be claimed to improve restored precision; for equal normalized-error bounds it worsens the absolute bound proportionally.'),
  limitations=['No keygen, CKKS context, public encoder or HE was run; scalar formulas were read from exact captured sources.',
   'The public float-scalar expression is evaluated at canonical scale and its adjacent binary64 scales; encoder rounding, rescale/key-switch errors and meta alignment are not bounded by this script.',
   'Real coefficient/message bounds and the bootstrap ApproxMod domain remain obligations. Base-capacity feasibility alone does not establish refresh precision.',
   'The input-level1 adjustment path matters. A direct base-level0 bootstrap that skips AdjustCiphertext would have different range accounting; this is not the BoundedSecureRefresh path.',
   'No result transfers from sparse period128 or p56/d4 to the new p60 common ring; the unchanged live proofs remain untouched.'],
  source_hashes_before=before)
 with tarfile.open(HERE/'sources.tar.gz','w:gz') as t:
  for p in paths:
   data=p.read_bytes();i=tarfile.TarInfo(str(p).lstrip('/'));i.size=len(data);t.addfile(i,io.BytesIO(data))
 after={str(p):sha(p.read_bytes()) for p in paths};assert before==after
 with tarfile.open(HERE/'sources.tar.gz') as t:assert {'/'+m.name:sha(t.extractfile(m).read()) for m in t.getmembers()}==before
 report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=sha((HERE/'sources.tar.gz').read_bytes()))
 (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(passed=True,profiles=rows),indent=2))
if __name__=='__main__':main()

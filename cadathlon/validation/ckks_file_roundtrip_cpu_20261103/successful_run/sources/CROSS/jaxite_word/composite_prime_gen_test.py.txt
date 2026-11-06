"""Prime searches must either return the prescribed chain or terminate."""
from absl.testing import absltest

from composite_prime_gen import composite_prime_gen


class CompositePrimeGenerationTest(absltest.TestCase):
  def test_exhausted_word_size_terminates(self):
    # Previously searched larger primes forever: none can fit 13 bits.
    with self.assertRaisesRegex(ValueError, 'within register_word_size'):
      composite_prime_gen(2, 16, 25, 24, 128, 13)

  def test_production_chain_is_unchanged(self):
    self.assertEqual(composite_prime_gen(2, 16, 45, 44, 32768, 32), [
        8257537, 1146881, 1179649, 1376257, 5308417, 5767169, 4882433,
        1769473, 4816897, 2424833, 2654209, 2752513, 4423681, 4620289,
        3604481, 3735553,
    ])


if __name__ == '__main__':
  absltest.main()

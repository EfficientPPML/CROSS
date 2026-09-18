// Independent OpenFHE v1.5.1 reference for retained-headroom demo profiles.
//
// This executable intentionally does not import CROSS parameter-generation
// code. It asks OpenFHE to generate each context, then checks and prints the
// physical Q/P layout that OpenFHE actually selected.

#include "openfhe.h"
#include "version.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lbcrypto;

namespace {

constexpr uint32_t kCompositeDegree = 2;
constexpr uint32_t kScalingModSize  = 60;
constexpr uint32_t kFirstModSize    = 61;
constexpr uint32_t kRegisterWordSize = 32;

struct Profile {
    const char* model;
    uint32_t logicalQ;
    uint32_t numLargeDigits;
};

template <typename Params>
std::vector<uint64_t> ModulusValues(const Params& params, size_t begin, size_t end) {
    std::vector<uint64_t> values;
    values.reserve(end - begin);
    for (size_t index = begin; index < end; ++index) {
        values.push_back(
            static_cast<uint64_t>(params[index]->GetModulus().ConvertToInt()));
    }
    return values;
}

template <typename Integer>
void PrintValues(const std::vector<Integer>& values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << values[index];
    }
    std::cout << ']';
}

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void Probe(const Profile& profile) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(profile.logicalQ - 1);
    parameters.SetScalingModSize(kScalingModSize);
    parameters.SetFirstModSize(kFirstModSize);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(profile.numLargeDigits);
    parameters.SetScalingTechnique(COMPOSITESCALINGMANUAL);
    parameters.SetCompositeDegree(kCompositeDegree);
    parameters.SetRegisterWordSize(kRegisterWordSize);

    const auto context = GenCryptoContext(parameters);
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(context->GetCryptoParameters());
    Require(cryptoParams != nullptr, std::string(profile.model) + ": CKKS parameter cast failed");
    Require(cryptoParams->GetCompositeDegree() == kCompositeDegree,
            std::string(profile.model) + ": unexpected composite degree");

    const auto& paramsQ  = cryptoParams->GetElementParams()->GetParams();
    const auto& paramsQP = cryptoParams->GetParamsQP()->GetParams();
    const size_t sizeQ   = paramsQ.size();
    const size_t sizeP   = paramsQP.size() - sizeQ;
    const auto qModuli   = ModulusValues(paramsQ, 0, sizeQ);
    const auto pModuli   = ModulusValues(paramsQP, sizeQ, paramsQP.size());
    const uint32_t logQP = cryptoParams->GetParamsQP()->GetModulus().GetMSB();

    Require(sizeQ == kCompositeDegree * profile.logicalQ,
            std::string(profile.model) + ": logical Q was not expanded by CD");
    for (uint64_t modulus : qModuli) {
        Require(modulus < (uint64_t{1} << 31),
                std::string(profile.model) + ": Q limb exceeds 31 bits");
    }
    for (uint64_t modulus : pModuli) {
        Require(modulus < (uint64_t{1} << 31),
                std::string(profile.model) + ": P limb exceeds 31 bits");
    }

    std::cout << "{\"model\":\"" << profile.model << "\","
              << "\"logical_q\":" << profile.logicalQ << ','
              << "\"num_large_digits\":" << profile.numLargeDigits << ','
              << "\"ring_n\":" << context->GetRingDimension() << ','
              << "\"physical_q\":" << sizeQ << ','
              << "\"physical_p\":" << sizeP << ','
              << "\"q_moduli\":";
    PrintValues(qModuli);
    std::cout << ",\"p_moduli\":";
    PrintValues(pModuli);
    std::cout << ",\"openfhe_log_qp\":" << logQP << "}\n";
}

}  // namespace

int main() {
    try {
        const std::string version = GetOPENFHEVersion();
        Require(version == "1.5.1", "probe requires OpenFHE 1.5.1, got " + version);
        std::cout << "{\"openfhe_version\":\"" << version
                  << "\",\"security\":\"HEStd_128_classic\","
                  << "\"composite_degree\":" << kCompositeDegree << ','
                  << "\"scaling_mod_size\":" << kScalingModSize << ','
                  << "\"first_mod_size\":" << kFirstModSize << ','
                  << "\"register_word_size\":" << kRegisterWordSize << "}\n";

        const std::vector<Profile> profiles{
            {"LoLAHE", 7, 4},
            {"LeNetHE", 9, 5},
            {"AlexNetTinyHE", 9, 5},
            {"AlexNetHE", 17, 9},
        };

        for (const auto& profile : profiles) {
            Probe(profile);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "OpenFHE reference probe failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

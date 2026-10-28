#include "Random.hpp"

#include "utilities/check.hpp"

namespace sfqmc::utils {

SeedType split_seed(int seed, boost::mpi3::communicator& comm, unsigned stream) {
  utils::check(stream < 256, "The stream index {} does not fit into a seed.", stream);
  utils::check(comm.rank() < (1 << 24), "A run of {} ranks does not fit into a seed.", comm.size());

  // splitmix64 finalizer. It is a bijection on 64 bits, so distinct (seed, rank, stream)
  // triples give distinct seeds, and flipping one bit of any of them changes about half of the
  // output bits. seed + rank would instead have neighbouring ranks start from neighbouring
  // states, and two runs whose seeds differ by one would share all but one of their streams.
  SeedType x = SeedType(unsigned(seed)) << 32 | SeedType(stream) << 24 | unsigned(comm.rank());
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

#if defined(ENABLE_DEVICE)
CurandRandomGenerator::CurandRandomGenerator(SeedType iseed) {
  cuda::curand_check(curandCreateGenerator(&handle_, CURAND_RNG_PSEUDO_MT19937),
                "curandCreateGenerator");
  cuda::curand_check(curandSetPseudoRandomGeneratorSeed(handle_, iseed),
                "curandSetPseudoRandomGeneratorSeed");
}

CurandRandomGenerator::CurandRandomGenerator(CurandRandomGenerator&& other) noexcept
  : handle_(other.handle_) {
  other.handle_ = nullptr;
}

CurandRandomGenerator& CurandRandomGenerator::operator=(CurandRandomGenerator&& other) noexcept {
  if(this != &other) {
    if(handle_ != nullptr) {
      curandDestroyGenerator(handle_);
    }
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

CurandRandomGenerator::~CurandRandomGenerator() {
  if(handle_ != nullptr) {
    curandDestroyGenerator(handle_);
  }
}
#endif

}

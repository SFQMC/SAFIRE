#include "Random.hpp"

namespace sfqmc::utils {

// Return Nth primer number
template<typename UInt>
UInt get_prime(UInt N)
{
  utils::check(not(N < 1),"N must be positive, provided N = {}", N);
  if(N==UInt(1)) return UInt(1);
  if(N==UInt(2)) return UInt(2);
  if(N==UInt(3)) return UInt(3);
  N-=UInt(3);
  std::vector<UInt> primes;
  primes.reserve(4096);
  primes.push_back(3);
  UInt largest = 3;
  while (N) { 
    largest += 2; 
    bool is_prime = true;
    for (int j = 0; j < primes.size(); j++) { 
      if (largest % primes[j] == 0) { 
        is_prime = false;
        break;
      }
      else if (primes[j] * primes[j] > largest) { 
        break;
      }
    }
    if (is_prime) { 
      primes.push_back(largest);
      N--;
    }
  }  
  return largest;
}

SeedType make_seed(boost::mpi3::communicator& comm) {
  // mpi3 has no datatype for SeedType, so the broadcast goes through unsigned long
  unsigned long baseoffset = 0;
  if(comm.root()) {
    baseoffset = static_cast<unsigned long>(std::time(0) % 1024);
  }
  comm.broadcast_value(baseoffset);
  baseoffset += static_cast<unsigned long>(comm.rank());
  return get_prime<SeedType>(SeedType(baseoffset));
}

SeedType split_seed(int seed, boost::mpi3::communicator& comm) {
  /*
  Splits the given seed across the given comm to guarantee that each MPI rank has a unique, but
    reproducible seed
  */
  SeedType baseoffset = seed;
  baseoffset += comm.rank();
  return get_prime<SeedType>(baseoffset);
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

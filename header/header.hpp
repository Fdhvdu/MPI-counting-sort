#ifndef HW1
#define HW1

//#define DO_NOT_OUTPUT	//do not output sorted file
//#define DEBUG_USE_COMM_TO_VALIDATE	//gather other processes' output to validate algorithm is correct
//#define MEASURE_TIME
//#define ONE_OUTPUT	//only one process will output the message, only useful when you turn on DEBUG_USE_COMM_TO_VALIDATE or MEASURE_TIME

#include<algorithm>	//min
#ifdef MEASURE_TIME
	#include<chrono>
#endif
#include<cstdint>
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
	#include<iosfwd>	//debug
#endif
#include<vector>
#include<mpi.h>

//I don't like std::int32_t and std::uint32_t
using size_type=std::uint32_t;	//[0,2147483647], 31 bit is enough
using value_type=std::int32_t;	//[-2147483647,2147483647]

class Comm;
template<class T>struct Spin_atomic;
using PV_buffer_t=std::vector<size_type>;

#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
extern std::stringstream ss;
#endif
#ifdef MEASURE_TIME
extern std::chrono::steady_clock::time_point begin_time;
extern std::vector<std::vector<std::chrono::microseconds::rep>> comm_time;	//for basic version
extern std::vector<std::chrono::microseconds::rep> allgather_time;	//for basic version
template<class T>
std::chrono::seconds::rep get_s(const T &val)
{
	return std::chrono::duration_cast<std::chrono::seconds>(val).count();
}
template<class T>
std::chrono::microseconds::rep get_microsecond(const T &val)
{
	return std::chrono::duration_cast<std::chrono::microseconds>(val).count();
}
template<class T>
std::chrono::microseconds::rep get_microsecond(const T &begin,const T &end)
{
	return get_microsecond(end-begin);
}
#endif

class CRange_alloc
{
public:
	using size_type=std::uint64_t;
	using value_type=std::int64_t;
private:
	size_type base_count_;	//every process has at least base_count of number to sort
	size_type residual_;	//some processes have additional number
public:
	CRange_alloc() noexcept;
	CRange_alloc(const size_type total_count,int process_count) noexcept;
	CRange_alloc(const CRange_alloc &)=default;
	size_type get_size(const int rank) const noexcept
	{
		return base_count_+(rank<residual_?1:0);
	}
	size_type get_offset(const int rank) const noexcept
	{
		return base_count_*rank+std::min<int>(residual_,rank);
	}
	CRange_alloc& operator=(const CRange_alloc &)=default;
};

constexpr auto TOTAL_MEM(103079215104);	//96 GiB
constexpr auto RESERVE_MEM(3221225472);	//3 GiB
constexpr auto USABLE_MEM(TOTAL_MEM-RESERVE_MEM);
constexpr CRange_alloc::size_type MAX_RANGE{4294967296};
constexpr size_type NUM_PER_THR{220000000};	//how many number can 1 thread deal with by using counting sort

void start(MPI_File ifile,MPI_File ofile,const Comm &comm,const CRange_alloc &alloc);
void start_advanced(size_type total_count,MPI_Comm mpi_comm,MPI_File ifile,MPI_File ofile,const Comm &comm,unsigned thr_count);

#endif
#include"../header/header.hpp"
#ifdef MEASURE_TIME
#include<algorithm>
#include<chrono>
#include<fstream>
#include<functional>
#include<iterator>
#include<numeric>
#endif
#include<stdexcept>	//runtime_error
#include<string>	//stoul
#ifdef MEASURE_TIME
#include<sstream>
#endif
#include<mpi.h>
#include"../header/common.hpp"
using namespace std;

//argv[1] is the number of int [0,2^31]
//argv[2] is input
//argv[3] is output
int main(int argc,char **argv)
{
#ifdef MEASURE_TIME
	ss<<"file name format: \"numbers to sort\"_\"MPI_Comm_rank(MPI_COMM_WORLD)\"_\"MPI_Comm_size(MPI_COMM_WORLD)\"_\"MPI_Get_processor_name\""<<endl;
	begin_time=chrono::steady_clock::now();
#endif
	if(argc!=4)
		throw runtime_error{"the number of arguments should be 3"};
	const size_type total_count(stol(argv[1]));
	if(total_count==0)
		return 0;
	check(MPI_Init(&argc,&argv));
	const int world_size{get_size()};
	const bool too_many_process{total_count<world_size};
	const int rank{get_rank()};
	if(rank==0||!too_many_process)
	{
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to open input file"<<endl;
#endif
		MPI_File ifile{open_r_file(too_many_process?MPI_COMM_SELF:MPI_COMM_WORLD,argv[2])};
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete open input file"<<endl;
#endif
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to open output file"<<endl;
#endif
		MPI_File ofile{open_w_file(too_many_process?MPI_COMM_SELF:MPI_COMM_WORLD,argv[3],total_count*4)};
		Comm comm{rank,too_many_process?1:world_size};
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to start"<<endl;
#endif
		start(ifile,ofile,comm,{total_count,comm.size});
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete start"<<endl;
#endif
		check(MPI_File_close(&ofile));
		check(MPI_File_close(&ifile));
	}
#ifdef MEASURE_TIME
	const string processor_name{get_processor_name()};
#endif
	check(MPI_Finalize());
#ifdef MEASURE_TIME
	const auto end_time(chrono::steady_clock::now()-begin_time);
	ss<<"total time: "<<get_s(end_time)<<"s "<<(get_microsecond(end_time)%1000000)<<"ms"<<endl;
	if(world_size!=1)
	{
		vector<chrono::milliseconds::rep> sum_up_comm_time;
		for(const auto &val:comm_time)
			sum_up_comm_time.emplace_back(accumulate(begin(val),end(val),0));
		ss<<"total send+recv communication: "<<accumulate(begin(sum_up_comm_time),end(sum_up_comm_time),0)<<"ms"<<endl;
		ss<<"total check communication: "<<accumulate(begin(allgather_time),end(allgather_time),0)<<"ms"<<endl;
		vector<chrono::milliseconds::rep> src(sum_up_comm_time.size());	//send+recv+check
		transform(begin(sum_up_comm_time),end(sum_up_comm_time),begin(allgather_time),begin(src),plus<chrono::milliseconds::rep>{});
		nth_element(begin(src),next(begin(src),src.size()/2),end(src));
		ss<<"min odd+even+check: "<<*min_element(begin(src),end(src))<<"ms"<<endl;
		ss<<"max odd+even+check: "<<*max_element(begin(src),end(src))<<"ms"<<endl;
		ss<<"mean odd+even+check: "<<static_cast<long double>(accumulate(begin(src),end(src),0))/src.size()<<"ms"<<endl;
		ss<<"median odd+even+check: "<<src[src.size()/2]<<"ms"<<endl;
	}
#ifdef ONE_OUTPUT
	if(rank==0)
	{
#endif
		ofstream ofs{to_string(total_count)+"_"+to_string(rank)+"_"+to_string(world_size)+"_"+processor_name};
		ofs<<ss.rdbuf();
#ifdef ONE_OUTPUT
	}
#endif
#endif
}
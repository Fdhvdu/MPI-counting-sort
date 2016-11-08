//#define LOCAL_DEBUG
#include"../header/header.hpp"
#include<algorithm>
#ifdef MEASURE_TIME
#include<chrono>
#endif
#if defined(MEASURE_TIME)||defined(DEBUG_USE_COMM_TO_VALIDATE)
#include<fstream>	//debug
#endif
#include<iostream>	//debug
#include<iterator>
#include<numeric>	//iota
#if defined(MEASURE_TIME)||defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(LOCAL_DEBUG)
#include<sstream>
#endif
#include<stdexcept>	//runtime_error
#include<string>
#include<thread>
#include<unordered_map>
#include<vector>
#include<mpi.h>
#include<sched.h>	//sched_getcpu
#include"../header/common.hpp"
using namespace std;

namespace
{
#ifdef LOCAL_DEBUG
	stringstream local_ss;
#endif
	int get_rank(const vector<int> &core_num_set,const int core_num,const string &processor_name_set,const string &find_name)
	{
		for(auto iter(begin(core_num_set));iter!=end(core_num_set);++iter)
			if(*iter==core_num)
			{
				const int rank(distance(begin(core_num_set),iter));
				if(equal(begin(find_name),end(find_name),next(begin(processor_name_set),rank*find_name.size())))
					return rank;
			}
		throw logic_error{"why cannot find rank"};
	}

	//vector<vector<rank of usable process>>
	vector<vector<int>> is_processor_header(const int core_num,const int world_size)
	{
		static constexpr int processor_name_length{4};
		vector<int> core_num_set(world_size);
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to allgather core_num"<<endl;
#endif
		allgather(&core_num,1,core_num_set.data());
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete allgather core_num"<<endl;
#endif
		char processor_name[MPI_MAX_PROCESSOR_NAME];
		int resultlen;
		check(MPI_Get_processor_name(processor_name,&resultlen));
		if(resultlen!=processor_name_length)
			throw logic_error{"size is not "+to_string(processor_name_length)};
		string processor_name_set(resultlen*world_size,0);
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to allgather processor_name"<<endl;
#endif
		allgather(processor_name,resultlen,&processor_name_set[0]);
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete allgather processor_name"<<endl;
#endif
		unordered_map<string,vector<int>> processor_name_core_num;
		for(size_t i{0};i!=processor_name_set.size();i+=resultlen)
			processor_name_core_num[processor_name_set.substr(i,resultlen)].emplace_back(core_num_set[i/resultlen]);
		//cat /proc/cpuinfo|grep "processor"
		//cat /proc/cpuinfo|grep "physical id"
		vector<vector<int>> result;
		for(const auto &val:processor_name_core_num)
		{
			const pair<vector<int>::const_iterator,vector<int>::const_iterator> minmax{minmax_element(begin(val.second),end(val.second))};
			result.emplace_back();
			if(*minmax.first<6)
				result.back().emplace_back(get_rank(core_num_set,*minmax.first,processor_name_set,val.first));
			if(5<*minmax.second)
				result.back().emplace_back(get_rank(core_num_set,*minmax.second,processor_name_set,val.first));
		}
		return result;
	}

	//pair<vector<rank>,vector<pair<rank,thr count>>>
	pair<vector<int>,vector<pair<int,unsigned>>> how_many_process_thr(const size_type total_count,const vector<vector<int>> &stat)
	{
		const unsigned max_thread_per_process{thread::hardware_concurrency()};
		vector<int> rank;
		for(auto iter(begin(stat));iter!=end(stat);++iter)
			if(iter->size()==1)
				rank.emplace_back(iter->front());
		for(size_t i{0};i!=2;++i)	//at most 2 processes for a node
			for(auto iter(begin(stat));iter!=end(stat);++iter)
				if(iter->size()==2)
					rank.emplace_back(iter->operator[](i));
		CRange_alloc apv(MAX_RANGE,rank.size());
#ifdef LOCAL_DEBUG
		local_ss<<"USABLE_MEM "<<USABLE_MEM<<" ("<<(USABLE_MEM>>30)<<" GiB)"<<endl;
		local_ss<<"file size "<<total_count*sizeof(size_type)<<" ("<<((total_count*sizeof(size_type))>>30)<<" GiB)"<<endl<<endl;
		for(const auto &val:stat)
		{
			local_ss<<"rank";
			for(size_t i{0};i!=val.size();++i)
				local_ss<<' '<<val[i];
			local_ss<<" USABLE_MEM - (file size * "<<val.size()<<") "<<(USABLE_MEM-total_count*sizeof(size_type)*val.size())<<" ("<<((USABLE_MEM-total_count*sizeof(size_type)*val.size())>>30)<<" GiB)"<<endl;
			local_ss<<"ask memory for pv_buf "<<(accumulate(begin(val),end(val),static_cast<remove_const<decltype(USABLE_MEM)>::type>(0),[&](const remove_const<decltype(USABLE_MEM)>::type init,const int val){return init+apv.get_size(distance(begin(rank),find(begin(rank),end(rank),val)));})*sizeof(size_type)/val.size())<<" ("<<((accumulate(begin(val),end(val),static_cast<remove_const<decltype(USABLE_MEM)>::type>(0),[&](const remove_const<decltype(USABLE_MEM)>::type init,const int val){return init+apv.get_size(distance(begin(rank),find(begin(rank),end(rank),val)));})*sizeof(size_type)/val.size())>>30)<<" GiB)"<<endl;
			local_ss<<"emplace_back "<<min<unsigned>((USABLE_MEM-total_count*sizeof(size_type)*val.size())/(accumulate(begin(val),end(val),static_cast<remove_const<decltype(USABLE_MEM)>::type>(0),[&](const remove_const<decltype(USABLE_MEM)>::type init,const int val){return init+apv.get_size(distance(begin(rank),find(begin(rank),end(rank),val)));})*sizeof(size_type)/val.size()),max_thread_per_process)<<" thread "<<endl<<endl;
		}
#endif
		//how many threads can a node use according to total_count
		//I think I don't have to sort node_thr (according to size of a node)
		vector<unsigned> node_thr;
		for(const auto &val:stat)
			node_thr.emplace_back(min<unsigned>((USABLE_MEM-total_count*sizeof(size_type)*val.size())/(accumulate(begin(val),end(val),static_cast<remove_const<decltype(USABLE_MEM)>::type>(0),[&](const remove_const<decltype(USABLE_MEM)>::type init,const int val){return init+apv.get_size(distance(begin(rank),find(begin(rank),end(rank),val)));})*sizeof(size_type)/val.size()),max_thread_per_process));	//no more than max_thread_per_process thread
		//schedule thread count
		//I assume all nodes have same architecture
		//so, maximum thread is rank.size()*max_thread_per_process
		auto require_thr(max(static_cast<unsigned>(1),min<unsigned>(total_count/NUM_PER_THR,rank.size()*max_thread_per_process)));
#ifdef LOCAL_DEBUG
		local_ss<<"require_thr "<<require_thr<<endl;
#endif
		decltype(require_thr) residual{0};
		//aptc=allocate process thread count
		CRange_alloc aptc(require_thr,node_thr.size());
		for(auto iter(begin(node_thr));iter!=end(node_thr);++iter)
		{
			const auto require_tc(aptc.get_size(distance(begin(node_thr),iter)));
			if(*iter<require_tc)
				residual+=require_tc-*iter;
		}
		for(auto iter(begin(node_thr));iter!=end(node_thr);++iter)
		{
			const auto require_tc(aptc.get_size(distance(begin(node_thr),iter)));
			if(require_tc<*iter)
				if(residual)
				{
					const auto gap(*iter-require_tc);
					if(gap<=residual)
						residual-=gap;
					else
					{
						*iter-=residual;
						residual=0;
					}
				}
				else
					*iter=require_tc;
		}
		//assign thread count
		vector<pair<int,unsigned>> vec;
		for(size_t i{0};i!=node_thr.size();++i)
			switch(stat[i].size())
			{
			case 1:
				vec.emplace_back(stat[i].front(),node_thr[i]);
				break;
			case 2:
				vec.emplace_back(stat[i].front(),node_thr[i]/2+(node_thr[i]%2));
				vec.emplace_back(stat[i].back(),node_thr[i]/2);
				break;
			default:
				throw logic_error{"why does a node have strange number of thread"};
			}
		//some process may get 0 thread
		vec.erase(remove_if(begin(vec),end(vec),[](const pair<int,unsigned> &val){return !val.second;}),end(vec));
		rank.erase(remove_if(begin(rank),end(rank),[&](const int rank){return find_if(begin(vec),end(vec),[=](const pair<int,unsigned> &val){return rank==val.first;})==end(vec);}),end(rank));
		return make_pair(move(rank),move(vec));
	}
}

//argv[1] is the number of integer [0,2^31]
//argv[2] is input
//argv[3] is output
int main(int argc,char **argv)
{
#ifdef MEASURE_TIME
	ss<<"file name format: \"numbers to sort\"_\"MPI_Comm_size(MPI_COMM_WORLD)\"_\"MPI_Get_processor_name\"_\"sched_getcpu()\"_\"rank in new comm\"_\"size in new comm\"_\"thread count\""<<endl;
#ifdef DEBUG_USE_COMM_TO_VALIDATE
	ss<<"DEBUG_USE_COMM_TO_VALIDATE"<<endl;
#endif
#ifdef ONE_OUTPUT
	ss<<"ONE_OUTPUT"<<endl;
#endif
	begin_time=chrono::steady_clock::now();
#endif
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
	ofstream ofs;
#endif
	if(argc!=4)
		throw runtime_error{"the number of arguments should be 3"};
	const size_type total_count(stol(argv[1]));
	if(total_count==0)
		return 0;
	//do not use this after MPI_Init_thread
	//MPI_Init_thread may change return value of sched_getcpu
	const int core_num{sched_getcpu()};
	{
		int provided;
		//actually, I don't care the return value of provided
		check(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided));
	}
	const int world_size{get_size()};
	const int world_rank{get_rank()};
	bool is_usable;
	pair<vector<int>,vector<pair<int,unsigned>>> process_thr;	//active process rank, thread count of process
	{
		const vector<vector<int>> stat{is_processor_header(core_num,world_size)};
		is_usable=find_if(begin(stat),end(stat),[=](const vector<int> &val){return find(begin(val),end(val),world_rank)!=end(val);})!=end(stat);
		if(is_usable)
		{
			process_thr=how_many_process_thr(total_count,stat);
			is_usable=(find(begin(process_thr.first),end(process_thr.first),world_rank)!=end(process_thr.first));
		}
	}
#ifdef LOCAL_DEBUG
	if(is_usable)	//check all processes get same process_thr
	{
		if(world_rank==process_thr.first.front())
		{
			for(size_t i{1};i!=process_thr.first.size();++i)
			{
				MPI_Status status;
				vector<int> rank_set(process_thr.first.size());
				check(MPI_Recv(rank_set.data(),rank_set.size(),MPI_INT,process_thr.first[i],0,MPI_COMM_WORLD,&status));
				check_count(status,MPI_INT);
				if(rank_set!=process_thr.first)
					cerr<<"process_thr.first of rank "<<process_thr.first[i]<<" is not equal to rank "<<process_thr.first.front()<<endl;
				for(size_t j{0};j!=process_thr.second.size();++j)
				{
					int pair_first;
					check(MPI_Recv(&pair_first,1,MPI_INT,process_thr.first[i],0,MPI_COMM_WORLD,&status));
					check_count(status,MPI_INT);
					unsigned pair_second;
					check(MPI_Recv(&pair_second,1,MPI_UNSIGNED,process_thr.first[i],0,MPI_COMM_WORLD,&status));
					check_count(status,MPI_UNSIGNED);
					if(process_thr.second[j].first!=pair_first)
						cerr<<"process_thr.second[j].first of rank "<<process_thr.first[i]<<" is not equal to rank "<<process_thr.first.front()<<endl;
					if(process_thr.second[j].second!=pair_second)
						cerr<<"process_thr.second[j].second of rank "<<process_thr.first[i]<<" is not equal to rank "<<process_thr.first.front()<<endl;
				}
			}
		}
		else
		{
			check(MPI_Ssend(process_thr.first.data(),process_thr.first.size(),MPI_INT,process_thr.first.front(),0,MPI_COMM_WORLD));
			for(size_t i{0};i!=process_thr.second.size();++i)
			{
				check(MPI_Ssend(&process_thr.second[i].first,1,MPI_INT,process_thr.first.front(),0,MPI_COMM_WORLD));
				check(MPI_Ssend(&process_thr.second[i].second,1,MPI_UNSIGNED,process_thr.first.front(),0,MPI_COMM_WORLD));
			}
		}
	}
#endif
#ifdef LOCAL_DEBUG
	if(is_usable&&world_rank==process_thr.first.front())
	{
		for(size_t i{0};i!=process_thr.first.size();++i)
		{
			if(i)
				local_ss<<' ';
			local_ss<<process_thr.first[i];
		}
		local_ss<<endl;
		for(size_t i{0};i!=process_thr.second.size();++i)
			local_ss<<"rank "<<process_thr.second[i].first<<" with "<<process_thr.second[i].second<<" thread"<<endl;
		local_ss<<endl;
	}
#endif
	MPI_Group world_group;
	MPI_Group new_group=MPI_GROUP_EMPTY;
	MPI_Comm new_comm;
	if(is_usable)
	{
		check(MPI_Comm_group(MPI_COMM_WORLD,&world_group));
		check(MPI_Group_incl(world_group,process_thr.first.size(),process_thr.first.data(),&new_group));
		check(MPI_Group_free(&world_group));
	}
	check(MPI_Comm_create(MPI_COMM_WORLD,new_group,&new_comm));
	int new_comm_rank,new_comm_size;
#ifdef ONE_OUTPUT
	bool I_am_rank_0{false};
#endif
	if(is_usable)
	{
		new_comm_rank=get_rank(new_comm);
		new_comm_size=get_size(new_comm);
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to open input file"<<endl;
#endif
		MPI_File ifile{open_r_file(new_comm,argv[2])};
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete open input file"<<endl;
#endif
		set_view(ifile,0);
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(DO_NOT_OUTPUT)
		MPI_File ofile;
#else
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to open output file"<<endl;
#endif
		MPI_File ofile{open_w_file(new_comm,argv[3])};
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete open output file"<<endl;
#endif
#endif
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to start_advanced"<<endl;
#endif
		start_advanced(total_count,new_comm,ifile,ofile,Comm(new_comm_rank,new_comm_size),find_if(begin(process_thr.second),end(process_thr.second),[=](const pair<int,unsigned> &val){return val.first==world_rank;})->second);
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete start_advanced"<<endl;
#endif
#if !(defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(DO_NOT_OUTPUT))
		check(MPI_File_close(&ofile));
#endif
		check(MPI_File_close(&ifile));
		check(MPI_Comm_free(&new_comm));
		check(MPI_Group_free(&new_group));
#ifdef ONE_OUTPUT
		if(new_comm_rank==0)
			I_am_rank_0=true;
#endif
	}
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
	const string processor_name{get_processor_name()};
#endif
	check(MPI_Finalize());
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_Finalize()"<<endl;
#endif
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
	if(is_usable)
	{
#ifdef ONE_OUTPUT
		if(I_am_rank_0)
		{
#endif
			ofs.open(to_string(total_count)+"_"+to_string(world_size)+"_"+processor_name+"_"+to_string(core_num)+"_"+to_string(new_comm_rank)+"_"+to_string(new_comm_size)+"_"+to_string(find_if(begin(process_thr.second),end(process_thr.second),[=](const pair<int,unsigned> &val){return val.first==world_rank;})->second));
#ifdef LOCAL_DEBUG
			ofs<<local_ss.rdbuf();
#endif
			ofs<<ss.rdbuf();
#ifdef ONE_OUTPUT
		}
#endif
	}
#endif
}
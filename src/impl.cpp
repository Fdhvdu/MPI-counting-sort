#include"../header/header.hpp"
#include<algorithm>
#include<atomic>
#include<array>
#ifdef MEASURE_TIME
#include<chrono>
#endif
#include<future>
#include<iterator>
#include<limits>	//numeric_limits
#include<memory>	//unique_ptr
#include<numeric>	//accumulate
#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
#include<sstream>
#endif
#include<vector>
#include<mpi.h>
#include"../header/common.hpp"
using namespace std;

#if defined(DEBUG_USE_COMM_TO_VALIDATE)||defined(MEASURE_TIME)
stringstream ss;
#endif
#ifdef MEASURE_TIME
chrono::steady_clock::time_point begin_time;
vector<vector<chrono::microseconds::rep>> comm_time;
vector<chrono::microseconds::rep> allgather_time;
#endif

//atb=alloccate thread buffer
//apv=alloccate process value
//atv=alloccate thread value
//tv=thread value (vector version)
//pv_begin=process value begin
//pv_end=process value end
namespace
{
	using Buffer_t=vector<value_type>;

	void advanced_sort(Buffer_t &,size_type,size_type
					   ,PV_buffer_t &,CRange_alloc::size_type
					   ,CRange_alloc::value_type,CRange_alloc::value_type
					   ,const CRange_alloc &,const vector<CRange_alloc::value_type> &
					   ,unsigned,unsigned
					   ,atomic<size_type> &,atomic<size_type> &
					   ,unique_ptr<Spin_atomic<value_type>[]> &
					   ,atomic<unsigned> &,atomic<bool> &
	);
	bool my_sort(Buffer_t::iterator,Buffer_t::iterator);
	void odd_even_sort(Buffer_t &,const Comm &,const CRange_alloc &);
	void odd_even_sort(Buffer_t &,const Comm &,bool,bool);
	void output_file(MPI_File,Buffer_t &,size_type);
	inline void output_file(MPI_File file,Buffer_t &buf)
	{
		output_file(file,buf,buf.size());
	}
	void output_file_nonblock(MPI_File,Buffer_t &,size_type);
	void output_file_nonblock_at(MPI_File,MPI_Offset,Buffer_t &,size_type);
	void read_file(MPI_File,Buffer_t &);
}

void start(MPI_File ifile,MPI_File ofile,const Comm &comm,const CRange_alloc &alloc)
{
	//to speed up, I use MPI_File_set_view
	//because I have already used MPI_File_set_view, I should use individual file pointer
	set_view(ifile,alloc.get_offset(comm.rank));
	Buffer_t number(alloc.get_size(comm.rank));
	read_file(ifile,number);
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to odd_even_sort"<<endl;
#endif
	odd_even_sort(number,comm,alloc);
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete odd_even_sort"<<endl;
#endif
	set_view(ofile,alloc.get_offset(comm.rank));	//this is bad, you should use set view early
	output_file(ofile,number);
}

void start_advanced(const size_type total_count,MPI_Comm mpi_comm,MPI_File ifile,MPI_File ofile,const Comm &comm,const unsigned thr_count)
{
	static constexpr auto min_val(numeric_limits<value_type>::min());
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to buf"<<endl;
#endif
	Buffer_t buf(total_count);	//buf.size(): [0,2^31]
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete buf"<<endl;
#endif
	CRead_MPI_File read_file(ifile,total_count);
	read_file.read(buf.data(),Get_MPI_Datatype<value_type>::value);
	CRange_alloc apv(MAX_RANGE,comm.size);
	const auto pv_buf_size(apv.get_size(comm.rank));
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to pv_buf"<<endl;
#endif
	vector<size_type> pv_buf(pv_buf_size*thr_count,0);
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete pv_buf"<<endl;
#endif
	const CRange_alloc atb(total_count,thr_count);
	const CRange_alloc::value_type pv_begin(min_val+apv.get_offset(comm.rank));
	const CRange_alloc::value_type pv_end(min_val+apv.get_offset(comm.rank+1));
	const CRange_alloc atv(pv_buf_size,thr_count);
	vector<future<void>> thr;
	thr.reserve(thr_count);	
	atomic<size_type> global_offset{0};
	atomic<size_type> global_count{0};	//[0,2^31]
	vector<CRange_alloc::value_type> tv(thr_count);
	for(unsigned i{0};i!=thr_count;++i)
		tv[i]=pv_begin+atv.get_offset(i);
	unique_ptr<Spin_atomic<value_type>[]> bucket{new Spin_atomic<value_type>[thr_count]};
	atomic<unsigned> complete{thr_count};
	atomic<bool> waiting{true};
	read_file.complete_read(buf.data(),Get_MPI_Datatype<value_type>::value);
#if 0
	ss<<"buf[0] "<<buf[0]<<endl
		<<"buf[1] "<<buf[1]<<endl
		<<"buf[buf.size()-2] "<<buf[buf.size()-2]<<endl
		<<"buf.back() "<<buf.back()<<endl;
#endif
#ifdef DEBUG_USE_COMM_TO_VALIDATE
	Buffer_t sort_buf;
	if(comm.rank==0)
	{
		sort_buf=buf;
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to sort sort_buf"<<endl;
#endif
		sort(begin(sort_buf),end(sort_buf));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete sort sort_buf"<<endl;
#endif
	}
#endif
#if 0
	//for debug
	for(size_t i{0};i!=buf.size();++i)
	{
		if(pv_begin<=buf[i])
		{
			if(buf[i]<pv_end)
			{
				++global_count;
				++pv_buf[(buf[i]-pv_begin)+0*pv_buf_size];
				++bucket[distance(begin(tv),prev(upper_bound(begin(tv),end(tv),buf[i])))].val;
			}
		}
		else
			++global_offset;
	}
#endif
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to counting sort"<<endl;
#endif
	if(1<thr_count)
	{
		for(unsigned i{0};i!=thr_count;++i)
			thr.emplace_back(async(launch::async,advanced_sort
							 ,ref(buf),atb.get_offset(i),atb.get_offset(i+1)
							 ,ref(pv_buf)
							 ,pv_buf_size
							 ,pv_begin,pv_end
							 ,ref(atv),ref(tv)
							 ,i,thr_count
							 ,ref(global_offset)
							 ,ref(global_count)
							 ,ref(bucket)
							 ,ref(complete),ref(waiting)
			));
		for(auto &val:thr)
			val.wait();
	}
	else
		advanced_sort(buf,0,total_count,pv_buf,pv_buf_size,pv_begin,pv_end,atv,tv,0,1,global_offset,global_count,bucket,complete,waiting);
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete counting sort"<<endl;
#endif
#if 0
	//for debug
	auto des(begin(buf));
	for(size_t i{0};i!=pv_buf_size;++i)
	{
		size_type val{0};
		for(unsigned j{0};j!=thr_count;++j)
			val+=pv_buf[i+j*pv_buf_size];
		if(val)
		{
			fill_n(des,val,static_cast<value_type>(i)+pv_begin);
			advance(des,val);
		}
	}
#endif
	const size_type write_count(global_count);
#ifdef DEBUG_USE_COMM_TO_VALIDATE
	const size_type trans_offset(global_offset);
	vector<size_type> offset_set(comm.size);
	if(1<comm.size)
		allgather(&trans_offset,1,offset_set.data(),mpi_comm);
	if(comm.rank==0)
	{
		auto des(buf.data());
		for(int i(1);i!=comm.size;++i)
			recv(next(des,offset_set[i]),offset_set[i]-offset_set[i-1],i,0,mpi_comm);
	}
	else
		check(MPI_Ssend(buf.data(),write_count,Get_MPI_Datatype<value_type>::value,0,0,mpi_comm));
	if(comm.rank==0)
		if(buf==sort_buf)
			ss<<"sort is successful"<<endl;
		else
			ss<<"sort is unsuccessful"<<endl;
#else
#ifndef DO_NOT_OUTPUT
	//do not use MPI_File_set_view and MPI_File_iwrite
	//because MPI_File_set_view is collective, it will wait other processes to call MPI_File_set_view
	//to solve this problem, I use MPI_File_iwrite_at
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to issue MPI_File_iwrite_at"<<endl;
#endif
	output_file_nonblock_at(ofile,global_offset,buf,write_count);
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_File_iwrite_at"<<endl;
#endif
#endif
#endif
}

namespace
{
	//pv_buf.size()!=pv_buf_size (pv_buf.size()==pv_buf_size*thr_count)
	//pv_buf_size: [1,2^32]
	//thr_num: [0,thr_count)
	//thr_count: [1,...
	//global_count: [0,2^32]
	void advanced_sort(Buffer_t &buf,const size_type thr_buf_begin,const size_type thr_buf_end
					   ,PV_buffer_t &pv_buf,const CRange_alloc::size_type pv_buf_size
					   ,const CRange_alloc::value_type pv_begin,const CRange_alloc::value_type pv_end
					   ,const CRange_alloc &atv,const vector<CRange_alloc::value_type> &tv
					   ,const unsigned thr_num,const unsigned thr_count
					   ,atomic<size_type> &global_offset,atomic<size_type> &global_count
					   ,unique_ptr<Spin_atomic<value_type>[]> &bucket
					   ,atomic<unsigned> &complete,atomic<bool> &waiting
	)
	{
		size_type local_offset{0};
		size_type local_count{0};
		vector<size_type> local_bucket(thr_count,0);
		const auto pv_buf_des(next(begin(pv_buf),pv_buf_size*thr_num));
		for(size_type i{thr_buf_begin};i!=thr_buf_end;++i)
			if(pv_begin<=buf[i])
			{
				if(buf[i]<pv_end)
				{
					++local_count;
					++(*next(pv_buf_des,buf[i]-pv_begin));
					++local_bucket[distance(begin(tv),prev(upper_bound(begin(tv),end(tv),buf[i])))];
				}
			}
			else
				++local_offset;
		for(unsigned i{0};i!=thr_count;++i)
			bucket[(thr_num+i)%thr_count].inc(local_bucket[(thr_num+i)%thr_count]);
		if((--complete)!=0)
			while(waiting)
				;
		else
#ifdef MEASURE_TIME
		{
			ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to waiting"<<endl;
#endif
			waiting=false;
#ifdef MEASURE_TIME
		}
#endif
		auto des(begin(buf));
		for(size_t i{0};i!=thr_num;++i)
			advance(des,bucket[i].val);
		auto count(bucket[thr_num].val);
		for(CRange_alloc::size_type i(atv.get_offset(thr_num));count;++i)
		{
			size_type val{0};
			CRange_alloc::size_type offset{0};
			for(unsigned j{0};j!=thr_count;++j,offset+=pv_buf_size)
				val+=pv_buf[i+offset];
			if(val)
			{
				fill_n(des,val,static_cast<value_type>(i)+pv_begin);
				advance(des,val);
				count-=val;
			}
		}
		global_offset+=local_offset;
		global_count+=local_count;
	}
	
	bool my_sort(Buffer_t::iterator begin,const Buffer_t::iterator end)
	{
		bool has_changed{false};
		while(begin!=end)
		{
			const Buffer_t::iterator next_=next(begin);
			if(*begin>*next_)
			{
				swap(*begin,*next_);
				has_changed=true;
			}
			advance(begin,2);
		}
		return has_changed;
	}

	void odd_even_sort(Buffer_t &number,const Comm &comm,const CRange_alloc &alloc)
	{
		//rank  0      1      2
		//index 0 1 2| 3 4 5| 6
		//for rank 1, number.size() is 3, begin is odd, end is even

		//rank  0      1    2
		//index 0 1 2| 3 4| 5
		//for rank 1, number.size() is 2, begin is odd, end is odd

		//rank  0    1      2
		//index 0 1| 2 3 4| 5
		//for rank 1, number.size() is 3, begin is even, end is odd

		//rank  0    1    2
		//index 0 1| 2 3| 4
		//for rank 1, number.size() is 2, begin is even, end is even

		//rank  0  1  2  3
		//index 0| 1| 2| 3
		//for rank 0, number.size() is 1, begin is even, end is odd
		//for rank 1, number.size() is 1, begin is odd, end is even

		const bool begin_is_odd(alloc.get_offset(comm.rank)&1);	//need to exchange data with previous rank when begin is odd
														//end_is_odd: need to exchange data with next rank when end is odd
		odd_even_sort(number,comm,begin_is_odd,begin_is_odd^(number.size()&1));
	}

	void odd_even_sort(Buffer_t &number,const Comm &comm,const bool begin_is_odd,const bool end_is_odd)
	{
#ifdef MEASURE_TIME
		vector<chrono::milliseconds::rep> local_comm_time;
		chrono::steady_clock::time_point local_begin_time;
#endif
		//0 is for even sort, 1 is for odd sort
		array<Buffer_t::iterator,2> first{{begin(number),next(begin(number))}},last{{end(number),prev(end(number))}};
		if(begin_is_odd)
			swap(first[0],first[1]);
		if(end_is_odd)
			swap(last[0],last[1]);
		bool again;
		do
		{
			again=false;
			for(char i{0};i!=2;++i)
			{
				future<bool> fut{async(launch::async,my_sort,first[i],last[i])};
				if(end_is_odd^i&&comm.rank!=comm.size-1)
				{
#ifdef MEASURE_TIME
					local_begin_time=chrono::steady_clock::now();
#endif
					check(MPI_Ssend(&number.back(),1,Get_MPI_Datatype<value_type>::value,comm.rank+1,0,MPI_COMM_WORLD));
#ifdef MEASURE_TIME
					local_comm_time.emplace_back(get_microsecond(local_begin_time,chrono::steady_clock::now()));
#endif
					value_type recv_val;
#ifdef MEASURE_TIME
					local_begin_time=chrono::steady_clock::now();
#endif
					recv(&recv_val,1,comm.rank+1,1);
#ifdef MEASURE_TIME
					local_comm_time.emplace_back(get_microsecond(local_begin_time,chrono::steady_clock::now()));
#endif
					if(recv_val<number.back())
					{
						number.back()=recv_val;
						again=true;
					}
				}
				if(begin_is_odd^i&&comm.rank!=0)
				{
					value_type recv_val;
#ifdef MEASURE_TIME
					local_begin_time=chrono::steady_clock::now();
#endif
					recv(&recv_val,1,comm.rank-1,0);
#ifdef MEASURE_TIME
					local_comm_time.emplace_back(get_microsecond(local_begin_time,chrono::steady_clock::now()));
#endif
#ifdef MEASURE_TIME
					local_begin_time=chrono::steady_clock::now();
#endif
					check(MPI_Ssend(number.data(),1,Get_MPI_Datatype<value_type>::value,comm.rank-1,1,MPI_COMM_WORLD));
#ifdef MEASURE_TIME
					local_comm_time.emplace_back(get_microsecond(local_begin_time,chrono::steady_clock::now()));
#endif
					if(recv_val>number[0])
					{
						number[0]=recv_val;
						again=true;
					}
				}
				again|=fut.get();
			}
			if(comm.size!=1)
			{
				bool *status=new bool[comm.size];
#ifdef MEASURE_TIME
				comm_time.emplace_back(move(local_comm_time));
				local_begin_time=chrono::steady_clock::now();
#endif
				check(MPI_Allgather(&again,1,MPI_CXX_BOOL,status,1,MPI_CXX_BOOL,MPI_COMM_WORLD));
#ifdef MEASURE_TIME
				allgather_time.emplace_back(get_microsecond(local_begin_time,chrono::steady_clock::now()));
#endif
				if(find(status,status+comm.size,true)==status+comm.size)
				{
					delete []status;
					break;
				}
				else
					again=true;
			}
		}while(again);
	}

	void output_file(MPI_File ofile,Buffer_t &buf,const size_type write_count)
	{
		//MPI_File_write_all_begin and MPI_File_write_all_end is better than MPI_File_write_all
		//however, I use MPI_File_write_all because nothing I can do (for this function) during begin to end
		//MPI_File_iwrite_all is not available for MPICH 3.1.4
		MPI_Status status;
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to MPI_File_write_all"<<endl;
#endif
		check(MPI_File_write_all(ofile,buf.data(),write_count,Get_MPI_Datatype<value_type>::value,&status));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_File_write_all"<<endl;
#endif
		check_count(status,Get_MPI_Datatype<value_type>::value);
	}

	void output_file_nonblock(MPI_File ofile,Buffer_t &buf,const size_type write_count)
	{
		MPI_Request request;
		check(MPI_File_iwrite(ofile,buf.data(),write_count,Get_MPI_Datatype<value_type>::value,&request));
		wait(request,Get_MPI_Datatype<value_type>::value);
	}

	void output_file_nonblock_at(MPI_File ofile,MPI_Offset offset,Buffer_t &buf,const size_type write_count)
	{
		MPI_Request request;
		check(MPI_File_iwrite_at(ofile,offset*sizeof(value_type),buf.data(),write_count,Get_MPI_Datatype<value_type>::value,&request));
		wait(request,Get_MPI_Datatype<value_type>::value);
	}
	
	void read_file(MPI_File ifile,Buffer_t &number)
	{
		//MPI_File_read_all_begin and MPI_File_read_all_end is better than MPI_File_read_all
		//however, I use MPI_File_read_all because nothing I can do (for this function) during begin to end
		//MPI_File_iread_all is not available for MPICH 3.1.4
		MPI_Status status;
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to MPI_File_read_all"<<endl;
#endif
		check(MPI_File_read_all(ifile,number.data(),number.size(),Get_MPI_Datatype<value_type>::value,&status));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_File_read_all"<<endl;
#endif
		check_count(status,Get_MPI_Datatype<value_type>::value);
	}
}
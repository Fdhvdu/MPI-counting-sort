#include"../header/common.hpp"
#include"../header/header.hpp"
#ifdef MEASURE_TIME
	#include<chrono>
#endif
#include<ostream>
#ifdef MEASURE_TIME
	#include<sstream>
#endif
#include<mpi.h>
using namespace std;

CRange_alloc::CRange_alloc() noexcept
	:base_count_{0},residual_{0}{}

CRange_alloc::CRange_alloc(const size_type total_count,const int process_count) noexcept
	:base_count_{total_count/process_count},residual_{total_count%process_count}{}

CRead_MPI_File::CRead_MPI_File(MPI_File file,const int count)
	:count_{count},file_{file}{}

void CRead_MPI_File::read(void *buf,MPI_Datatype datatype)
{
	if(MPI_FILE_IREAD_BEGIN<=count_&&count_<MPI_FILE_IREAD_END)
#ifdef MEASURE_TIME
	{
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" to issue MPI_File_iread"<<endl;
#endif
		check(MPI_File_iread(file_,buf,count_,datatype,&request_));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" to complete issue MPI_File_iread"<<endl;
	}
#endif
	else
#ifdef MEASURE_TIME
	{
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to MPI_File_read"<<endl;
#endif
		check(MPI_File_read(file_,buf,count_,datatype,&status_));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_File_read"<<endl;
	}
#endif
}

void CRead_MPI_File::complete_read(void *buf,MPI_Datatype datatype)
{
	if(MPI_FILE_IREAD_BEGIN<=count_&&count_<MPI_FILE_IREAD_END)
#ifdef MEASURE_TIME
	{
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to MPI_Wait"<<endl;
#endif
		check(MPI_Wait(&request_,&status_));
#ifdef MEASURE_TIME
		ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_Wait"<<endl;
	}
#endif
	check_count(status_,datatype);
}

Comm::Comm(const int rank_,const int size_) noexcept
	:rank{rank_},size{size_}{}

#ifndef USELESS_CHECK
void check(const int status)
{
	if(status)
		MPI_Abort(MPI_COMM_WORLD,status);
}

void check_count(MPI_Status status,MPI_Datatype datatype)
{
	int count;
	check(MPI_Get_count(&status,datatype,&count));
	check(count==MPI_UNDEFINED);
}
#endif

string get_processor_name()
{
	string name(MPI_MAX_PROCESSOR_NAME,0);
	int resultlen;
	check(MPI_Get_processor_name(&name[0],&resultlen));
	name.resize(resultlen);
	return name;
}

int get_rank(MPI_Comm comm)
{
	int rank;
	check(MPI_Comm_rank(comm,&rank));
	return rank;
}

int get_size(MPI_Comm comm)
{
	int size;
	check(MPI_Comm_size(comm,&size));
	return size;
}

MPI_File open_r_file(MPI_Comm comm,const string &file_name)
{
	MPI_File file;
	//MPI_Info_set access_style, collective_buffering, nb_proc
	check(MPI_File_open(comm,file_name.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file));
	return file;
}

MPI_File open_w_file(MPI_Comm comm,const string &file_name)
{
	MPI_File file;
	check(MPI_File_open(comm,file_name.c_str(),MPI_MODE_WRONLY|MPI_MODE_CREATE|MPI_MODE_UNIQUE_OPEN,MPI_INFO_NULL,&file));
	return file;
}

MPI_File open_w_file(MPI_Comm comm,const string &file_name,MPI_Offset size)
{
	MPI_File file{open_w_file(comm,file_name)};
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete open output file"<<endl;
#endif
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to MPI_File_preallocate"<<endl;
#endif
	check(MPI_File_preallocate(file,size));
#ifdef MEASURE_TIME
	ss<<get_microsecond(begin_time,chrono::steady_clock::now())<<" microsecond to complete MPI_File_preallocate"<<endl;
#endif
	return file;
}

void set_view(MPI_File file,MPI_Offset disp)
{
	check(MPI_File_set_view(file,disp*sizeof(value_type),Get_MPI_Datatype<value_type>::value,Get_MPI_Datatype<value_type>::value,"native",MPI_INFO_NULL));
}

void wait(MPI_Request request,MPI_Datatype datatype)
{
	MPI_Status status;
	check(MPI_Wait(&request,&status));
	check_count(status,datatype);
}
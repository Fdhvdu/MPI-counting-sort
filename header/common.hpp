#ifndef COMMON
#define COMMON
#include<algorithm>	//min
#include<atomic>
#include<cstdint>
#include<string>
#include<type_traits>
#include<mpi.h>

//#define USELESS_CHECK	//turn on means "do not check return value of MPI function"

class CRead_MPI_File
{
	static constexpr auto MPI_FILE_IREAD_BEGIN=1048576;	//use MPI_File_iread if read count is bigger or equal to MPI_FILE_IREAD_BEGIN
	static constexpr auto MPI_FILE_IREAD_END=536870912;
	const int count_;
	MPI_File file_;
	MPI_Request request_;
	MPI_Status status_;
public:
	CRead_MPI_File(MPI_File file,int count);
	void read(void *buf,MPI_Datatype datatype);
	void complete_read(void *buf,MPI_Datatype datatype);
};

struct Comm
{
	const int rank;
	const int size;
	Comm(int rank,int size) noexcept;
};

template<class T>
struct Spin_atomic
{
	T val;
	Spin_atomic() noexcept
		:val{0},flag_{ATOMIC_FLAG_INIT}{}
	void inc() noexcept
	{
		while(flag_.test_and_set())
			;
		++val;
		flag_.clear();
	}
	void inc(const T val_) noexcept
	{
		while(flag_.test_and_set())
			;
		val+=val_;
		flag_.clear();
	}
private:
	std::atomic_flag flag_;
};

template<class T>
struct Get_MPI_Datatype;

#ifdef USELESS_CHECK
inline void check(int) noexcept{};
inline void check_count(MPI_Status,MPI_Datatype) noexcept{};
#else
void check(int);	//check return value of MPI functions
void check_count(MPI_Status,MPI_Datatype);
#endif
std::string get_processor_name();
int get_rank(MPI_Comm=MPI_COMM_WORLD);
int get_size(MPI_Comm=MPI_COMM_WORLD);
MPI_File open_r_file(MPI_Comm comm,const std::string &file_name);
MPI_File open_w_file(MPI_Comm comm,const std::string &file_name);
MPI_File open_w_file(MPI_Comm comm,const std::string &file_name,MPI_Offset size);
void set_view(MPI_File,MPI_Offset);
void wait(MPI_Request,MPI_Datatype);

template<class T>
void recv(T des,const int count,const int source,const int tag,MPI_Comm comm=MPI_COMM_WORLD)
{
	static constexpr auto TYPE(Get_MPI_Datatype<typename std::remove_pointer<T>::type>::value);
	MPI_Status status;
	check(MPI_Recv(des,count,TYPE,source,tag,comm,&status));
	check_count(status,TYPE);
}

//1. sendcount must be the same for comm
//2. you have to guarantee the size of recvbuf is at least sendcount*get_size(comm)
template<class InIter,class OutInter>
void allgather(InIter sendbuf,const int sendcount,OutInter recvbuf,MPI_Comm comm=MPI_COMM_WORLD)
{
	using namespace std;
	static constexpr auto send_recv_type(Get_MPI_Datatype<typename remove_cv<typename remove_pointer<InIter>::type>::type>::value);
	static constexpr int tag{0};
	const int rank{get_rank(comm)};
	const auto rank_begin(recvbuf+rank*sendcount);
	copy(sendbuf,sendbuf+sendcount,rank_begin);
	const int size{get_size(comm)};
	if(size==1)
		return ;
	const auto rank_end(recvbuf+(rank+1)*sendcount);
	const int recvbuf_size{size*sendcount};
	const auto recvbuf_end(recvbuf+recvbuf_size);
	MPI_Status status;
	//this is a very dirty code, make it beautiful in the future
	if(rank==0||rank==size-1)
	{
		if(rank)	//end
			check(MPI_Sendrecv(
				rank_begin,
				sendcount,
				send_recv_type,rank-1,tag,
				recvbuf,
				recvbuf_size-sendcount,
				send_recv_type,rank-1,tag,
				comm,&status
			));
		else
			check(MPI_Sendrecv(
				rank_begin,
				sendcount,
				send_recv_type,rank+1,tag,
				rank_end,
				recvbuf_size-sendcount,
				send_recv_type,rank+1,tag,
				comm,&status
			));
		check_count(status,send_recv_type);
	}
	else
	{
		const int middle{size/2};
		if(rank!=middle)
		{
			if(middle<rank)	//right
			{
				check(MPI_Recv(
					rank_end,
					recvbuf_end-rank_end,
					send_recv_type,rank+1,tag,comm,&status
				));
				check_count(status,send_recv_type);
				check(MPI_Sendrecv(
					rank_begin,
					recvbuf_end-rank_begin,
					send_recv_type,rank-1,tag,
					recvbuf,
					rank_begin-recvbuf,
					send_recv_type,rank-1,tag,
					comm,&status
				));
				check_count(status,send_recv_type);
				check(MPI_Ssend(
					recvbuf,
					rank_end-recvbuf,
					send_recv_type,rank+1,tag,comm
				));
			}
			else	//left
			{
				check(MPI_Recv(
					recvbuf,
					rank_begin-recvbuf,
					send_recv_type,rank-1,tag,comm,&status
				));
				check_count(status,send_recv_type);
				check(MPI_Sendrecv(
					recvbuf,
					rank_end-recvbuf,
					send_recv_type,rank+1,tag,
					rank_end,
					recvbuf_end-rank_end,
					send_recv_type,rank+1,tag,
					comm,&status
				));
				check_count(status,send_recv_type);
				check(MPI_Ssend(
					rank_begin,
					recvbuf_end-rank_begin,
					send_recv_type,rank-1,tag,comm
				));
			}
		}
		else
		{
			check(MPI_Recv(
				rank_end,
				recvbuf_end-rank_end,
				send_recv_type,rank+1,tag,comm,&status
			));	//right
			check_count(status,send_recv_type);
			check(MPI_Recv(
				recvbuf,
				rank_begin-recvbuf,
				send_recv_type,rank-1,tag,comm,&status
			));	//left
			check_count(status,send_recv_type);
			check(MPI_Ssend(
				recvbuf,
				rank_end-recvbuf,
				send_recv_type,rank+1,tag,comm
			));	//right
			check(MPI_Ssend(
				rank_begin,
				recvbuf_end-rank_begin,
				send_recv_type,rank-1,tag,comm
			));	//left
		}
	}
}

template<>
struct Get_MPI_Datatype<char>
{
	static constexpr auto value=MPI_CHAR;
};

template<>
struct Get_MPI_Datatype<int>
{
	static constexpr auto value=MPI_INT;
};

template<>
struct Get_MPI_Datatype<unsigned>
{
	static constexpr auto value=MPI_UNSIGNED;
};

template<>
struct Get_MPI_Datatype<long>
{
	static constexpr auto value=MPI_LONG;
};

template<>
struct Get_MPI_Datatype<unsigned long>
{
	static constexpr auto value=MPI_UNSIGNED_LONG;
};

#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef unsigned int uint;
typedef unsigned long long int ulong;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

//__global__ void addKernel(int* c, const int* a, const int* b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}

__device__ __host__ uint TausStep(uint, int, int, int, uint);
__device__ __host__ uint LCGStep(uint z, uint A, uint C);

typedef struct RandomResult
{
	int x;
	int y;
	int z;
	int w;
	float value;
} RandomResult;

typedef struct Possibilities
{
	int origins[32];
	uint boards[32];
	uint colors[32];
	uint kings[32];
	int wins[32];
};

__device__ __host__ RandomResult Random(RandomResult);

__device__ __host__ void DisplayBoard(char board[8][8]);

__device__ __host__ void Encode(char board[8][8], uint*, uint*, uint*);
__device__ __host__ void Decode(char board[8][8], uint, uint, uint);

__device__ __host__ void PrintBits(uint);

__device__ __host__ void Move(uint*, uint*, uint*, int, int);
__device__ __host__ void Remove(uint*, uint*, int);

__global__ void SimulateGame(RandomResult*, Possibilities*, int);
__device__ __host__ int MakeMove(uint*, uint*, uint*, RandomResult*, bool);

__device__ __host__ bool MultipleHit(uint*, uint*, uint*, RandomResult*, int*);
__device__ __host__ int CalculateScore(uint, uint, int);

int FindPossibleMoves(uint, uint, uint, int, Possibilities*);
bool FindPossibleMultipleHit(uint, uint, uint, int, int, int, int*, Possibilities*, int);
void AddPossible(uint, uint, uint, int, int, int, int, Possibilities*);
void DisplayPossibilities(Possibilities, int);


int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	Possibilities possibilities;
	for (int i = 0; i < 32; i++)
	{
		possibilities.origins[i] = 0;
		possibilities.boards[i] = 0;
		possibilities.colors[i] = 0;
		possibilities.kings[i] = 0;
		possibilities.wins[i] = 0;
	}

	srand(time(NULL));
	RandomResult random;
	random.x = rand() + 128;
	random.y = rand() + 128;
	random.z = rand() + 128;
	random.w = rand();

	char checkersBoard[8][8];
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			checkersBoard[i][j] = '0';
		}
	}
	//DisplayBoard(checkersBoard);
	//printf("\n\n\n");
	for (int i = 0; i < 3; i++)
	{
		for (int j = 1 - i % 2; j < 8; j += 2)
		{
			checkersBoard[i][j] = 'Y';
			checkersBoard[7 - i][7 - j] = 'X';
		}
	}
	DisplayBoard(checkersBoard);
	//printf("\n\n\n");
	uint occupied = 0, color = 0, kings = 0;
	/*occupied |= 1 << 18;
	occupied |= 1 << 17;
	color |= 1 << 18;
	kings |= 1 << 18;
	Decode(checkersBoard, occupied, color, kings);
	DisplayBoard(checkersBoard);
	MakeMove(&occupied, &color, &kings, &random, 0);
	Decode(checkersBoard, occupied, color, kings);
	DisplayBoard(checkersBoard);*/

	Encode(checkersBoard, &occupied, &color, &kings);
	PrintBits(occupied);
	PrintBits(color);
	PrintBits(kings);
	int whites = 12, blacks = 12;
	int result;
	int moves = 0;
	int maxMoves = 0;
	for (int j = 0; j < 0; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				checkersBoard[i][j] = '0';
			}
		}
		for (int i = 0; i < 3; i++)
		{
			for (int j = 1 - i % 2; j < 8; j += 2)
			{
				checkersBoard[i][j] = 'Y';
				checkersBoard[7 - i][7 - j] = 'X';
			}
		}
		occupied = 0;
		color = 0;
		kings = 0;

		Encode(checkersBoard, &occupied, &color, &kings);
		for (int i = 0; i < 100; i++)
		{
			//DisplayPossibilities(possibilities, moves);
			if (moves > maxMoves) maxMoves = moves;
			//printf("%d possible moves \n", moves);
			result = MakeMove(&occupied, &color, &kings, &random, i % 2);
			whites >>= 5;whites = CalculateScore(occupied, color, i%2);
			blacks = whites & 31;
			whites >>= 5;
			if (result == -1)
			{
				/*printf("Koniec gry!\n");
				printf("Tury: %d Biali: %d Czarni: %d\n", i, whites, blacks);
				if (i % 2)
				{
					printf("Czarni nie moga wykonac ruchu!\n");
				}
				else
				{
					printf("Biali nie moga wykonac ruchu!\n");
				}
				Decode(checkersBoard, occupied, color, kings);
				DisplayBoard(checkersBoard);*/
				break;
			}
			else if (result > 0)
			{
				if (whites == 0)
				{
					printf("Biali przegrali!\n");
					Decode(checkersBoard, occupied, color, kings);
					DisplayBoard(checkersBoard);
					break;
				}

				//blacks -= result;
				if (blacks == 0)
				{
					printf("Czarni przegrali\n");
					Decode(checkersBoard, occupied, ~color, kings);
					DisplayBoard(checkersBoard);
					break;
				}
			}
			if (i % 2)
			{
				Decode(checkersBoard, occupied, ~color, kings);
				printf("Y\n");
			}
			else
			{
				Decode(checkersBoard, occupied, color, kings);
				printf("X\n");
			}
			printf("Biali: %d Czarni: %d\n", whites, blacks);
			DisplayBoard(checkersBoard);
			color = ~color;
		}
	}
	printf("\n Max ruchow: %d \n", maxMoves);



	RandomResult* d_random = 0;
	uint* d_board = 0;
	uint* d_colors = 0;
	uint* d_kings = 0;
	Possibilities* d_possibilities = 0;

	cudaError_t cudaStatus;

	int blocks;
	//DisplayPossibilities(possibilities, blocks);
	//for (int i = 0; i < blocks; i++)
	//{
	//	printf("%d\n", possibilities.boards[i]);
	//}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaEvent_t m_start, m_stop;
	cudaEventCreate(&m_start);
	cudaEventCreate(&m_stop);
	
	cudaEventRecord(m_start);
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_random, sizeof(RandomResult));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_possibilities, sizeof(Possibilities));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.


	cudaEventRecord(m_stop);
	cudaEventSynchronize(m_stop);
	float m_seconds = 0;
	cudaEventElapsedTime(&m_seconds, m_start, m_stop);
	m_seconds /= 1000;
	printf("Alokowanie pamieci zajelo %f sekund\n", m_seconds);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int maxI, max;
	
	cudaEventRecord(start);
	printf("Poczatek symulacji...\n");
	for (int i = 0; i < 3; i++)
	{
		blocks = FindPossibleMoves(occupied, color, kings, i%2, &possibilities);
		printf("%d mozliwosci\n", blocks);
		if (i) DisplayPossibilities(possibilities, blocks);

		cudaStatus = cudaMemcpy(d_random, &random, sizeof(RandomResult), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(d_possibilities, &possibilities, sizeof(Possibilities), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		dim3 blocks3(blocks, 10, 1);
		SimulateGame << <blocks3, 1000 >> > (d_random, d_possibilities, (i + 1) % 2);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		random.w += rand() % 100;
		random.x += rand() % 100;
		random.y += rand() % 100;
		random.z += rand() % 100;

		cudaStatus = cudaMemcpy(&possibilities, d_possibilities, sizeof(Possibilities), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		maxI = 0;
		max = possibilities.wins[0];
		for (int i = 1; i < blocks; i++)
		{
			if (possibilities.wins[i] > max)
			{
				maxI = i;
				max = possibilities.wins[i];
			}
		}
		for (int i = 0; i < blocks; i++) printf("Blok %d: %d\n", i, possibilities.wins[i]);
		printf("Najlepsza mozliwosc %d\n", maxI + 1);
		occupied = possibilities.boards[maxI];
		color = possibilities.colors[maxI];
		kings = possibilities.kings[maxI];
		occupied = possibilities.boards[1];
		color = possibilities.colors[1];
		kings = possibilities.kings[1];
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	miliseconds /= 1000;
	printf("Symulacja zakonczona po %f sekundach", miliseconds);

	printf("\n");


	  // Add vectors in parallel.
	  //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	  //if (cudaStatus != cudaSuccess) {
	  //    fprintf(stderr, "addWithCuda failed!");
	  //    return 1;
	  //}

	  //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	  //    c[0], c[1], c[2], c[3], c[4]);

	  //// cudaDeviceReset must be called before exiting in order for profiling and
	  //// tracing tools such as Nsight and Visual Profiler to show complete traces.
	  //cudaStatus = cudaDeviceReset();
	  //if (cudaStatus != cudaSuccess) {
	  //    fprintf(stderr, "cudaDeviceReset failed!");
	  //    return 1;
	  //}
Error:
	cudaFree(d_random);
	cudaFree(d_possibilities);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.

__global__ void SimulateGame(RandomResult* o_random, Possibilities* o_possibilites, int turn)
{
	RandomResult m_random = *o_random;
	m_random.w += threadIdx.x * (o_possibilites->boards[blockIdx.x] >> 26);
	m_random.x += blockIdx.x * (o_possibilites->boards[blockIdx.x] >> 26);
	m_random.y += blockIdx.y * (o_possibilites->boards[blockIdx.x] >> 26);
	m_random.z += o_possibilites->origins[blockIdx.x] * (o_possibilites->boards[blockIdx.x] >> 26);
	uint m_board = o_possibilites->boards[blockIdx.x];
	uint m_colors = o_possibilites->colors[blockIdx.x];
	uint m_kings = o_possibilites->kings[blockIdx.x];
	int whites = CalculateScore(m_board, m_colors, turn);
	int blacks = whites & 31;
	whites >>= 5;
	RandomResult* random = &m_random;
	uint* board = &m_board;
	uint* colors = &m_colors;
	uint* kings = &m_kings;
	

	char checkersBoard[8][8];
	//Decode(checkersBoard, *board, *colors, *kings);
	//DisplayBoard(checkersBoard);
	//PrintBits(m_board);
	//printf("%d\n", m_board);

	int result;
	for (int i = turn; i < 100; i++)
	{
		result = MakeMove(board, colors, kings, random, i % 2);
		if (result == -1)
		{
			/*printf("Koniec gry!\n");
			printf("Tury: %d Biali: %d Czarni: %d\n", i, whites, blacks);
			if (i % 2)
			{
				printf("Czarni nie moga wykonac ruchu!\n");
			}
			else
			{
				printf("Biali nie moga wykonac ruchu!\n");
			}*/
			//Decode(checkersBoard, occupied, color, kings);
			//DisplayBoard(checkersBoard);
			break;
		}
		else if (result > 0)
		{
			if (i % 2)
			{
				whites -= result;
				if (whites == 0)
				{
					/*printf("Biali przegrali!\n");
					Decode(checkersBoard, *board, ~(*colors), *kings);
					DisplayBoard(checkersBoard);*/
					break;
				}
			}
			else
			{
				blacks -= result;
				if (blacks == 0)
				{
					/*printf("Czarni przegrali\n");
					Decode(checkersBoard, *board, *colors, *kings);
					DisplayBoard(checkersBoard);*/
					break;
				}
			}
		}
		/*if (i % 2)
		{
			Decode(checkersBoard, *board, ~(*colors), *kings);
			printf("Y\n");
		}
		else
		{
			Decode(checkersBoard, *board, *colors, *kings);
			printf("X\n");
		}
		printf("Biali: %d Czarni: %d\n", whites, blacks);
		DisplayBoard(checkersBoard);*/
		*colors = ~(*colors);
	}
	if (turn) blacks = __syncthreads_count(whites);
	else blacks = __syncthreads_count(blacks);
	if (threadIdx.x == 0)
	{
		printf("Nasi wygrali %d gier w bloku %d\n", blacks, blockIdx.x);
		//o_possibilites->wins[blockIdx.x] += blacks;
		atomicAdd(&(o_possibilites->wins[blockIdx.x]), blacks);
	}
	//PrintBits(*board);
	//PrintBits(*colors);
	//PrintBits(*kings);
}
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
//{
//	int* dev_a = 0;
//	int* dev_b = 0;
//	int* dev_c = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Allocate GPU buffers for three vectors (two input, one output)    .
//	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Launch a kernel on the GPU with one thread for each element.
//	addKernel << <1, size >> > (dev_c, dev_a, dev_b);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_c);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//
//	return cudaStatus;
//}

__device__ __host__ uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);
}

__device__ __host__ uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);
}

__device__ __host__ RandomResult Random(RandomResult random)
{
	RandomResult result;
	result.x = TausStep(random.x, 13, 19, 12, 4294967294);
	result.y = TausStep(random.y, 2, 25, 4, 4294967288);
	result.z = TausStep(random.z, 3, 11, 17, 4294967280);
	result.w = LCGStep(random.w, 1664525, 1013904223);

	result.value = 2.3283064365387e-10 * (result.x ^ result.y ^ result.z ^ result.w); // -0.5 0.5

	return result;
}

__device__ __host__ void DisplayBoard(char board[8][8])
{
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			printf("%c", board[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

__device__ __host__ void Encode(char board[8][8], uint* occupied, uint* color, uint* kings)
{
	*occupied = 0;
	*color = 0;
	*kings = 0;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (board[i][j] != '0')
			{
				*occupied |= 1 << (i * 4 + j / 2);
				if (board[i][j] == 'X')
				{
					*color |= 1 << (i * 4 + j / 2);
				}
				else if (board[i][j] == 'x')
				{
					*color |= 1 << (i * 4 + j / 2);
					*kings |= 1 << (i * 4 + j / 2);
				}
				else if (board[i][j] == 'y')
				{
					*kings |= 1 << (i * 4 + j / 2);
				}
			}
		}
	}
}

__device__ __host__ void Decode(char board[8][8], uint occupied, uint color, uint kings)
{
	int k;
	for (int i = 0; i < 8; i++)
	{
		k = i % 2;
		for (int j = 0; j < 4; j++)
		{
			board[i][2 * j + k] = '0';
			if (occupied & 1)
			{
				if (color & 1)
				{
					if (kings & 1)
					{
						board[i][2 * j + 1 - k] = 'x';
					}
					else board[i][2 * j + 1 - k] = 'X';
				}
				else
				{
					if (kings & 1)
					{
						board[i][2 * j + 1 - k] = 'y';
					}
					else board[i][2 * j + 1 - k] = 'Y';
				}
			}
			else
			{
				board[i][2 * j + 1 - k] = '0';
			}
			occupied >>= 1;
			color >>= 1;
			kings >>= 1;
		}
	}
}

void PrintBits(uint s)
{
	for (int i = 0; i < 32; i++)
	{
		printf("%d", s >> (31 - i) & 1);
	}
	printf("\n");
}



__device__ __host__ void Move(uint* board, uint* color, uint* kings, int n, int d)
{
	(*board) &= ~(1 << n);
	(*board) |= 1 << (n + d);
	(*color) |= 1 << (n + d);
	if (*kings & (1 << n))
	{
		(*kings) &= ~(1 << n);
		(*kings) |= 1 << (n + d);
	}
}

__device__ __host__ void Remove(uint* board, uint* kings, int n)
{
	(*board) &= ~(1 << n);
	(*kings) &= ~(1 << n);
}

void TempRemove(uint* board, int n)
{
	(*board) &= ~(1 << n);
}

void Add(uint* board, int n)
{
	(*board) |= 1 << n;
}

__device__ __host__ int MakeMove(uint* occupied, uint* color, uint* kings, RandomResult* random, bool flag) // zakladam ze rusza sie tylko kolor oznaczony 1
{
	*random = Random(*random);
	uint t_occupied = (*occupied) & (*color);
	uint enemies = (*occupied) & ~(*color);
	int k, n = 0;
	int count = 0;
	bool killing = false;
	int t_n;

	n = 0;
	while (t_occupied > 0) // liczenie mozliwych ruchow
	{
		if (t_occupied & 1) // na n-tym miejscu jest pionek dobrego koloru
		{
			k = (n >> 2) & 1; // (n / 4) % 2
			if ((n >> 2) > 0) // drugi rzad, moze w gore
			{
				if ((n & 3) + 1 - k) // druga kolumna, moze w lewo
				{
					if (enemies & (1 << n - 4 - k)) // zajete
					{
						if ((n >> 2) > 1 && (n & 3) > 0) // podwojny skos w lewo gora
						{
							if (!((*occupied) & (1 << n - 9))) //podwójny skos mozliwy
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, -9);
								killing = true;
							}
						}
					}
					else if ((*kings & (1 << n) || !flag) && !((*occupied) & (1 << n - 4 - k))) // wolne lewo gora
					{
						count++;
						if (*kings & (1 << n))
						{
							t_n = n - 4 -k;
							k ^= 1;
							while ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
							{
								t_n += -4 - k;
								k ^= 1;
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n - 4 - k)))
							{
								//printf("morderczy krol\n");
								killing = true;
							}
							k = (n >> 2) & 1;
						}
						//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, -4 - k);
					}
				}
				if ((n & 3) - k < 3) // przedostatnia kolumna, moze w prawo
				{
					if (enemies & (1 << n - 3 - k)) // zajete
					{
						if (((n >> 2) > 1) && ((n & 3) < 3)) // podwojny skos prawo gora
						{
							if (!((*occupied) & (1 << n - 7))) // podwojny skos mozliwy
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, -7);
								killing = true;
							}
						}
					}
					else if ((*kings & (1 << n) || !flag) && !((*occupied) & (1 << n - 3 - k))) // wolne prawo gora
					{
						count++;
						if (*kings & (1 << n))
						{
							t_n = n - 3 - k;
							k ^= 1;
							while ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
							{
								t_n += -3 - k;
								k ^= 1;
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n - 3 - k)))
							{
								//printf("Morderczy krol\n");
								killing = true;
							}
							k = (n >> 2) & 1;
						}
						//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, -3 - k);
					}
				}
			} // opcje w gore sprawdzone
			if ((n >> 2) < 7) // przedostatni rzad, mozna w dol
			{
				if ((n & 3) + 1 - k > 0) // druga kolumna, mozna w prawo
				{
					if (enemies & (1 << n + 4 - k)) // zajete
					{
						if (((n >> 2) < 6) && ((n & 3) > 0)) // podwojny skos lewo dol
						{
							if (!((*occupied) & (1 << n + 7))) // podwojny skos mozliwy
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, 7);
								killing = true;
							}
						}
					}
					else if ((*kings & (1 << n) || flag) && !((*occupied) & (1 << n + 4 - k)))// wolne lewo dol
					{
						count++;
						if (*kings & (1 << n))
						{
							t_n = n + 4 - k;
							k ^= 1;
							while ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
							{
								t_n += 4 - k;
								k ^= 1;
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n + 4 - k)))
							{
								//printf("Morderczy krol\n");
								killing = true;
							}
							k = (n >> 2) & 1;
						}
						//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, 4 - k);
					}
				}
				if (((n & 3) - k) < 3) // przedostatnia kolumna, mozna w prawo
				{
					if (enemies & (1 << n + 5 - k)) // zajete
					{
						if (((n >> 2) < 6) && ((n & 3) < 3)) // podwojny skos prawo dol
						{
							if (!((*occupied) & (1 << n + 9))) // podwojny skos mozliwy
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, 9);
								killing = true;
							}
						}
					}
					else if ((*kings & (1 << n) || flag) && !((*occupied) & (1 << n + 5 - k))) // wolne prawo dol
					{
						count++;
						if (*kings & (1 << n))
						{
							t_n = n + 5 - k;
							k ^= 1;
							while ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
							{
								t_n += 5 - k;
								k ^= 1;
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n + 5 - k)))
							{
								//printf("Morderczy krol\n");
								killing = true;
							}
							k = (n >> 2) & 1;
						}
						//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, 5 - k);
					}
				}
			} // opcje w dol sprawdzone
		}
		n++;
		t_occupied >>= 1;
	}

	//printf("Znaleziono %d ruchow\n", count);

	if (count == 0)
	{
		//printf("Nie można wykonać ruchu!\n");
		return -1;
	}
	//if (killing) printf("Musze bic!\n");
	t_occupied = (*occupied) & (*color);
	int moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
	//int moves = 1;
	count = 0;


	while (1) // Wykonywanie ruchu
	{
		n = 0;
		while (t_occupied > 0)
		{
			if (t_occupied & 1) // na n-tym miejscu jest pionek dobrego koloru
			{
				k = (n >> 2) & 1; // (n / 4) % 2
				if ((n >> 2) > 0) // drugi rzad, moze w gore
				{
					if ((n & 3) + 1 - k) // druga kolumna, moze w lewo
					{
						if (enemies & (1 << n - 4 - k)) // zajete
						{
							if ((n >> 2) > 1 && (n & 3) > 0) // podwojny skos w lewo gora
							{
								if (!((*occupied) & (1 << n - 9))) //podwójny skos mozliwy
								{
									count++;
									//printf("Znaleziono %d ruch %d -> %d lglg\n", count, n, -9);
									if (count == moves)
									{
										//printf("Wykonany ruch!\n");
										//printf("Bicie! %d -> %d\n", n, -9);
										Remove(occupied,kings, n - 4 - k);
										Move(occupied, color, kings, n, -9);
										count = 1;
										n -= 9;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										if ((n >> 2) == flag * 7)
										{
											*kings |= 1 << n;
										}
										return count;
									}
								}
							}
						}
						else if (!((*occupied) & (1 << n - 4 - k))) // wolne lewo gora
						{
							if (killing && (*kings & (1 << n)))
							{
								t_n = n - 4 - k;
								k ^= 1;
								while ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
								{
									t_n += -4 - k;
									k ^= 1;
								}
								if ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n - 4 - k)))
								{
									count++;
									if (count == moves)
									{
										//printf("Krol atakuje! %d -> %d\n", n, t_n - n - 4 - k);
										Move(occupied, color, kings, n, t_n - n - 4 - k);
										Remove(occupied, kings, t_n);
										count = 1;
										n = t_n - 4 - k;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										/*PrintBits(*occupied);
										PrintBits(*color);*/
										return count;
									}
								}
								k = (n >> 2) & 1;
							}
							if (!killing && *kings & (1 << n))
							{
								count++;
								if (count == moves)
								{
									count = 1;
									t_n = n - 4 - k;
									k ^= 1;
									while ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
									{
										count++;
										t_n += -4 - k;
										k ^= 1;
									}
									if (*occupied & (1 << t_n)) count--;
									moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
									//printf("Wykrylem %d ruchow i wykonuje %d ruch\n", count, moves);
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * (-9) + (moves & 1) * (-4 - k));
									return 0;
								}
							}
							else if (!killing && !flag)
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lg\n", count, n, -4 - k);
								if (count == moves)
								{
									//printf("Wykonany ruch!\n");
									Move(occupied, color, kings, n, -4 - k);
									if (((n - 4 - k) >> 2) == 0) *kings |= 1 << n - 4 - k;
									return 0;
								}
							}
						}
					}
					if ((n & 3) - k < 3) // przedostatnia kolumna, moze w prawo
					{
						if (enemies & (1 << n - 3 - k)) // zajete
						{
							if (((n >> 2) > 1) && ((n & 3) < 3)) // podwojny skos prawo gora
							{
								if (!((*occupied) & (1 << n - 7))) // podwojny skos mozliwy
								{
									count++;
									//printf("Znaleziono %d ruch %d -> %d pgpg\n", count, n, -7);
									if (count == moves)
									{
										//printf("Wykonany ruch!\n");
										//printf("Bicie! %d -> %d\n", n, -7);
										Remove(occupied, kings,n - 3 - k);
										Move(occupied, color, kings, n, -7);
										count = 1;
										n -= 7;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										if ((n >> 2) == flag * 7) *kings |= 1 << n;
										return count;
									}
								}
							}
						}
						else if (!((*occupied) & (1 << n - 3 - k))) // wolne prawo gora
						{
							if (killing && (*kings & (1 << n)))
							{
								t_n = n - 3 - k;
								k ^= 1;
								while ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
								{
									t_n += -3 - k;
									k ^= 1;
								}
								if ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n - 3 - k)))
								{
									count++;
									if (count == moves)
									{
										//printf("Krol atakuje! %d -> %d\n", n, t_n - n - 4 - k);
										Move(occupied, color, kings, n, t_n - n - 3 - k);
										Remove(occupied, kings, t_n);
										count = 1;
										n = t_n - 3 - k;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
								k = (n >> 2) & 1;
							}
							if (!killing && *kings & (1 << n))
							{
								count++;
								if (count == moves)
								{
									count = 1;
									t_n = n - 3 - k;
									k ^= 1;
									while ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
									{
										count++;
										t_n += -3 - k;
										k ^= 1;
									}
									if (*occupied & (1 << t_n)) count--;
									moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
									//printf("Wykrylem %d ruchow i wykonuje %d ruch\n", count, moves);
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * (-7) + (moves & 1) * (-3 - k));
									return 0;
								}
								k = (n >> 2) & 1;
							}
							else if (!killing && !flag)
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d pg\n", count, n, -3 -k);
								if (count == moves)
								{
									//printf("Wykonany ruch!\n");
									Move(occupied, color, kings, n, -3 - k);
									if ((n - 3 - k) >> 2 == 0) *kings |= 1 << n - 3 - k;
									return 0;
								}

							}
						}
					}
				} // opcje w gore sprawdzone
				if ((n >> 2) < 7) // przedostatni rzad, mozna w dol
				{
					if ((n & 3) + 1 - k > 0) // druga kolumna, mozna w prawo
					{
						if (enemies & (1 << n + 4 - k)) // zajete
						{
							if (((n >> 2) < 6) && ((n & 3) > 0)) // podwojny skos lewo dol
							{
								if (!((*occupied) & (1 << n + 7))) // podwojny skos mozliwy
								{
									count++;
									//printf("Znaleziono %d ruch %d -> %d ldld\n", count, n, 7);
									if (count == moves)
									{
										//printf("Wykonany ruch!\n");
										//printf("Bicie! %d -> %d\n", n, 7);
										Remove(occupied, kings, n + 4 - k);
										Move(occupied, color, kings, n, 7);
										count = 1;
										n += 7;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										if (n >> 2 == flag * 7) *kings |= 1 << n;
										return count;
									}
								}
							}
						}
						else if (!((*occupied) & (1 << n + 4 - k)))// wolne lewo dol
						{
							if (killing && (*kings & (1 << n)))
							{
								t_n = n + 4 - k;
								k ^= 1;
								while ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
								{
									t_n += 4 - k;
									k ^= 1;
								}
								if ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n + 4 - k)))
								{
									count++;
									if (count == moves)
									{
										//printf("Krol atakuje! %d -> %d\n", n, t_n - n - 4 - k);
										Move(occupied, color, kings, n, t_n - n + 4 - k);
										Remove(occupied, kings, t_n);
										n = t_n + 4 - k;
										count = 1;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
								k = (n >> 2) & 1;
							}
							if (!killing && *kings & (1 << n))
							{
								count++;
								if (count == moves)
								{
									count = 1;
									t_n = n + 4 - k;
									k ^= 1;
									while ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(*occupied & (1 << t_n)))
									{
										count++;
										t_n += 4 - k;
										k ^= 1;
									}
									if (*occupied & (1 << t_n)) count--;
									moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
									//printf("Wykrylem %d ruchow i wykonuje %d ruch\n", count, moves);
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * 7 + (moves & 1) * (4 - k));
									return 0;
								}
							}
							else if (!killing && flag)
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d ld\n", count, n, 4 - k);
								if (count == moves)
								{
									//printf("Wykonany ruch!\n");
									//printf("Wykonany ruch!\n");
									Move(occupied, color, kings, n, 4 - k);
									if ((n + 4 - k) >> 2 == 7) *kings |= 1 << n + 4 - k;
									return 0;
								}
							}
						}
					}
					if (((n & 3) - k) < 3) // przedostatnia kolumna, mozna w prawo
					{
						if (enemies & (1 << n + 5 - k)) // zajete
						{
							if (((n >> 2) < 6) && ((n & 3) - k < 3)) // podwojny skos prawo dol
							{
								if (!((*occupied) & (1 << n + 9))) // podwojny skos mozliwy
								{
									count++;
									//printf("Znaleziono %d ruch %d -> %d pdpd\n", count, n, 9);
									if (count == moves)
									{
										//printf("Wykonany ruch!\n");
										//printf("Bicie! %d -> %d\n", n, 9);
										Remove(occupied, kings, n + 5 - k);
										Move(occupied, color, kings, n, 9);
										count = 1;
										n += 9;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										if (n >> 2 == flag * 7) *kings |= 1 << n;
										return count;
									}
								}
							}
						}
						else if (!((*occupied) & (1 << n + 5 - k))) // wolne
						{
							if (killing && (*kings & (1 << n)))
							{
								t_n = n + 5 - k;
								k ^= 1;
								while ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
								{
									t_n += 5 - k;
									k ^= 1;
								}
								if ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(*occupied & (1 << t_n + 5 - k)))
								{
									count++;
									if (count == moves)
									{
										//printf("Krol atakuje! %d -> %d\n", n, t_n - n - 4 - k);
										Move(occupied, color, kings, n, t_n - n + 5 - k);
										Remove(occupied, kings, t_n);
										n = t_n + 5 - k;
										count = 1;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
								k = (n >> 2) & 1;
							}
							if (!killing && *kings & (1 << n))
							{
								count++;
								if (count == moves)
								{
									count = 1;
									t_n = n + 5 - k;
									k ^= 1;
									while ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(*occupied & (1 << t_n)))
									{
										count++;
										t_n += 5 - k;
										k ^= 1;
									}
									if (*occupied & (1 << t_n)) count--;
									moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
									//printf("Wykrylem %d ruchow i wykonuje %d ruch\n", count, moves);
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * 9 + (moves & 1) * (5 - k));
									return 0;
								}
							}
							if (!killing && flag)
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d pd\n", count, n, 5 -k);
								if (count == moves)
								{
									//printf("Wykonany ruch!\n");
									Move(occupied, color, kings, n, 5 - k);
									if ((n + 5 - k) >> 2 == 7) *kings |= 1 << n + 5 - k;
									return 0;
								}
							}
						}
					}
				} // opcje w dol sprawdzone
			}
			n++;
			t_occupied >>= 1;
		}
		t_occupied = (*occupied) & (*color);
	}
}

__device__ __host__ bool MultipleHit(uint* occupied, uint* color, uint* kings, RandomResult* random, int* n)
{
	int count = 0;
	int k = (*n >> 2) & 1;
	*random = Random(*random);
	int moves = (int)((random->value + 0.5) / 0.25) + 1;
	uint enemies = *occupied & ~(*color);
	while (1)
	{
		if ((*n >> 2) > 1) // mozna dwa razy w gore
		{
			if ((*n & 3) > 0) // mozna dwa razy w lewo
			{
				if ((enemies & (1 << *n - 4 - k)) && !((*occupied) & (1 << *n - 9))) // raz zajete dwa wolne
				{
					count++;
					if (count == moves)
					{
						//printf("Kontynuuje bicie! %d -> %d\n", *n, -9);
						Move(occupied, color, kings, *n, -9);
						Remove(occupied, kings, *n - 4 - k);
						*n = *n - 9;
						return true;
					}
				}
			}
			if ((*n & 3) < 3) // mozna dwa razy w prawo
			{
				if ((enemies & (1 << *n - 3 - k)) && !((*occupied) & (1 << *n - 7))) // raz zajete dwa wolne
				{
					count++;
					if (count == moves)
					{
						//printf("Kontynuuje bicie! %d -> %d\n", *n, -7);
						Move(occupied, color, kings, *n, -7);
						Remove(occupied, kings, *n - 3 - k);
						*n = *n - 7;
						return true;
					}
				}
			}
		}
		if ((*n >> 2) < 6) // mozna dwa razy w dol
		{
			if ((*n & 3) > 0) // mozna dwa razy w lewo
			{
				if ((enemies & (1 << *n + 4 - k)) && !((*occupied) & (1 << *n + 7))) // raz zajete dwa wolne
				{
					count++;
					if (count == moves)
					{
						//printf("Kontynuuje bicie! %d -> %d\n", *n, 7);
						Move(occupied, color, kings, *n, 7);
						Remove(occupied, kings, *n + 4 - k);
						*n = *n + 7;
						return true;
					}
				}
			}
			if ((*n & 3) < 3) // mozna dwa razy w prawo
			{
				if ((enemies & (1 << *n + 5 - k)) && !((*occupied) & (1 << *n + 9))) // raz zajete dwa wolne
				{
					count++;
					if (count == moves)
					{
						//printf("Kontynuuje bicie! %d -> %d\n", *n, 9);
						Move(occupied, color, kings, *n, 9);
						Remove(occupied, kings, *n + 5 - k);
						*n = *n + 9;
						return true;
					}
				}
			}
		}

		if (count == 0) // koniec bicia
		{
			//printf("Koniec bicia!\n");
			return false;
		}
	}
}

int FindPossibleMoves(uint occupied, uint color, uint kings, int flag, Possibilities* possibilities)
{
	int possible = 0;
	int k, n = 0;
	int t_n;
	uint t_occupied = occupied & color;
	uint enemies = occupied & ~color;
	while (t_occupied > 0) // liczenie mozliwych ruchow
	{
		if (t_occupied & 1) // na n-tym miejscu jest pionek dobrego koloru
		{
			k = (n >> 2) & 1; // (n / 4) % 2
			if ((n >> 2) > 0) // drugi rzad, moze w gore
			{
				if ((n & 3) + 1 - k) // druga kolumna, moze w lewo
				{
					if (enemies & (1 << n - 4 - k)) // zajete
					{
						if ((n >> 2) > 1 && (n & 3) > 0) // podwojny skos w lewo gora
						{
							if (!((occupied) & (1 << n - 9))) //podwójny skos mozliwy
							{
								//printf("Znaleziono %d -> lglg", n);
								if (FindPossibleMultipleHit(occupied, color, kings, n, -9, n - 4 - k, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, -9, n - 4 - k, possible, possibilities);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || !flag) && !((occupied) & (1 << n - 4 - k))) // wolne lewo gora
					{
						AddPossible(occupied, color, kings, n, -4 - k, -1, possible, possibilities);
						possible++;
						//printf("Znaleziono %d -> lg", n);
						if (kings & (1 << n))
						{
							t_n = n - 4 - k;
							k ^= 1;
							while ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities);
								possible++;
								//printf("Znaleziono krol %d -> lg", t_n + 4 + 1 - k);
								t_n += -4 - k;
								k ^= 1;
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n - 4 - k)))
							{
								//printf("Znaleziono krol %d -> lglg", t_n + 4 + 1 - k);
								Move(&occupied, &color, &kings, n, t_n + 4 + 1 - k);
								TempRemove(&occupied, t_n);
								if (FindPossibleMultipleHit(occupied, color, kings, t_n + 4 + 1 - k, -4 - k, t_n, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, t_n - 4 - k - n, t_n, possible, possibilities);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n + 4 + 1 - k, n);
								Add(&occupied, t_n);
							}
							k = (n >> 2) & 1;
						}
					}
				}
				if ((n & 3) - k < 3) // przedostatnia kolumna, moze w prawo
				{
					if (enemies & (1 << n - 3 - k)) // zajete
					{
						if (((n >> 2) > 1) && ((n & 3) < 3)) // podwojny skos prawo gora
						{
							if (!((occupied) & (1 << n - 7))) // podwojny skos mozliwy
							{
								//printf("Znaleziono %d -> pgpg", n);
								if (FindPossibleMultipleHit(occupied, color, kings, n, -7, n - 3 - k, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, -7, n - 3 - k, possible, possibilities);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || !flag) && !((occupied) & (1 << n - 3 - k))) // wolne prawo gora
					{
						AddPossible(occupied, color, kings, n, -3 - k, -1, possible, possibilities);
						possible++;
						//printf("Znaleziono %d -> pg", n);
						if (kings & (1 << n))
						{
							t_n = n - 3 - k;
							k ^= 1;
							while ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities);
								t_n += -3 - k;
								k ^= 1;
								possible++;
								//printf("Znaleziono krol %d -> pg", t_n + 3 + 1 - k);
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n - 3 - k)))
							{
								//printf("Znaleziono krol %d -> pgpg", t_n + 3 + 1 - k);
								Move(&occupied, &color, &kings, n, t_n + 3 + 1 - k);
								TempRemove(&occupied, t_n);
								if (FindPossibleMultipleHit(occupied, color, kings, t_n + 3 + 1 - k, -3 - k, t_n, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, t_n - 3 - k - n, t_n, possible, possibilities);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n + 3 + 1 - k, n);
								Add(&occupied, t_n);
							}
							k = (n >> 2) & 1;
						}
					}
				}
			} // opcje w gore sprawdzone
			if ((n >> 2) < 7) // przedostatni rzad, mozna w dol
			{
				if ((n & 3) + 1 - k > 0) // druga kolumna, mozna w prawo
				{
					if (enemies & (1 << n + 4 - k)) // zajete
					{
						if (((n >> 2) < 6) && ((n & 3) > 0)) // podwojny skos lewo dol
						{
							if (!((occupied) & (1 << n + 7))) // podwojny skos mozliwy
							{
								//printf("Znaleziono %d -> ldld", n);
								if (FindPossibleMultipleHit(occupied, color, kings, n, 7, n + 4 - k, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, 7, n + 4 - k, possible, possibilities);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || flag) && !((occupied) & (1 << n + 4 - k)))// wolne lewo dol
					{
						AddPossible(occupied, color, kings, n, 4 - k, -1, possible, possibilities);
						possible++;
						//printf("Znaleziono %d -> ld", n);
						if (kings & (1 << n))
						{
							t_n = n + 4 - k;
							k ^= 1;
							while ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities);
								t_n += 4 - k;
								k ^= 1;
								possible++;
								//printf("Znaleziono krol %d -> ld", t_n - 4 + 1 - k);
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n + 4 - k)))
							{
								//printf("Znaleziono %d -> ldld", t_n - 4 + 1 - k);
								Move(&occupied, &color, &kings, n, t_n - 4 + 1 - k);
								TempRemove(&occupied, t_n);
								if (FindPossibleMultipleHit(occupied, color, kings, t_n - 4 + 1 - k, 4 - k, t_n, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, t_n + 3 - k - n, t_n, possible, possibilities);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n - 4 + 1 - k, n);
								Add(&occupied, t_n);
							}
							k = (n >> 2) & 1;
						}
					}
				}
				if (((n & 3) - k) < 3) // przedostatnia kolumna, mozna w prawo
				{
					if (enemies & (1 << n + 5 - k)) // zajete
					{
						if (((n >> 2) < 6) && ((n & 3) < 3)) // podwojny skos prawo dol
						{
							if (!((occupied) & (1 << n + 9))) // podwojny skos mozliwy
							{
								//printf("Znaleziono %d -> pdpd", n);
								if (FindPossibleMultipleHit(occupied, color, kings, n, 9, n + 5 - k, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, 9, n + 5 - k, possible, possibilities);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || flag) && !((occupied) & (1 << n + 5 - k))) // wolne prawo dol
					{
						AddPossible(occupied, color, kings, n, 5 - k, -1, possible, possibilities);
						possible++;
						//printf("Znaleziono %d -> pd", n);
						if (kings & (1 << n))
						{
							t_n = n + 5 - k;
							k ^= 1;
							while ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities);
								t_n += 5 - k;
								k ^= 1;
								possible++;
								//printf("Znaleziono krol %d -> pd", t_n - 5 + 1 - k);
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n + 5 - k)))
							{
								//printf("Znaleziono krol %d -> pdpd", t_n - 5 + 1 - k);
								Move(&occupied, &color, &kings, n, t_n - 5 + 1 - k);
								TempRemove(&occupied, t_n);
								if (FindPossibleMultipleHit(occupied, color, kings, t_n - 5 + 1 - k, 5 - k, t_n, &possible, possibilities, n))
								{
									AddPossible(occupied, color, kings, n, t_n + 5 - k - n, t_n, possible, possibilities);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n - 5 + 1 - k, n);
								Add(&occupied, t_n);
							}
							k = (n >> 2) & 1;
						}
					}
				}
			} // opcje w dol sprawdzone
		}
		n++;
		t_occupied >>= 1;
	}
	return possible;
}

bool FindPossibleMultipleHit(uint occupied, uint color, uint kings, int n, int move, int hit, int* possible, Possibilities* possibilities, int origin)
{
	bool end = true;
	Move(&occupied, &color, &kings, n, move);
	Remove(&occupied, &kings, hit);
	n += move;
	int count = 0;
	int k = (n >> 2) & 1;
	uint enemies = occupied & ~(color);
	if ((n >> 2) > 1) // mozna dwa razy w gore
	{
		if ((n & 3) > 0) // mozna dwa razy w lewo
		{
			if ((enemies & (1 << n - 4 - k)) && !((occupied) & (1 << n - 9))) // raz zajete dwa wolne
			{
				end = false;
				//printf("Kontynuacja bicia %d -> lglg", n);
				if (FindPossibleMultipleHit(occupied, color, kings, n, -9, n - 4 - k, possible, possibilities, origin))
				{
					AddPossible(occupied, color, kings, origin, -9, n - 4 - k, *possible, possibilities);
					(*possible)++;
				}
			}
		}
		if ((n & 3) < 3) // mozna dwa razy w prawo
		{
			if ((enemies & (1 << n - 3 - k)) && !((occupied) & (1 << n - 7))) // raz zajete dwa wolne
			{
				end = false;
				//printf("Kontynuacja bicia %d -> pgpg", n);
				if (FindPossibleMultipleHit(occupied, color, kings, n, -7, n - 3 - k, possible, possibilities, n))
				{
					AddPossible(occupied, color, kings, origin, -7, n - 3 - k, *possible, possibilities);
					(* possible)++;
				}
			}
		}
	}
	if ((n >> 2) < 6) // mozna dwa razy w dol
	{
		if ((n & 3) > 0) // mozna dwa razy w lewo
		{
			if ((enemies & (1 << n + 4 - k)) && !((occupied) & (1 << n + 7))) // raz zajete dwa wolne
			{
				end = false;
				//printf("Kontynuacja bicia %d -> ldld", n);
				if (FindPossibleMultipleHit(occupied, color, kings, n, 7, n + 4 - k, possible, possibilities, n))
				{
					AddPossible(occupied, color, kings, origin, 7, n + 4 - k, *possible, possibilities);
					(*possible)++;
				}
			}
		}
		if ((n & 3) < 3) // mozna dwa razy w prawo
		{
			if ((enemies & (1 << n + 5 - k)) && !((occupied) & (1 << n + 9))) // raz zajete dwa wolne
			{
				end = false;
				//printf("Kontynuacja bicia %d -> pdpd", n);
				if (FindPossibleMultipleHit(occupied, color, kings, n, 9, n + 5 - k, possible, possibilities, n))
				{
					AddPossible(occupied, color, kings, origin, 9, n + 5 - k, *possible, possibilities);
					(* possible)++;
				}
			}
		}
	}
	return end;
}

void AddPossible(uint occupied, uint color, uint kings, int n, int move, int hit, int possibleIndex, Possibilities* possibilities)
{
	if (move != 0) Move(&occupied, &color, &kings, n, move);
	if (hit >= 0) Remove(&occupied, &kings, hit);
	possibilities->origins[possibleIndex] = n;
	possibilities->boards[possibleIndex] = occupied;
	possibilities->colors[possibleIndex] = ~color;
	possibilities->kings[possibleIndex] = kings;
	possibilities->wins[possibleIndex] = 0;
}

void DisplayPossibilities(Possibilities possibilities, int count)
{
	char board[8][8];
	for (int i = 0; i < count; i++)
	{
		printf("Mozliwosc z %d\n", possibilities.origins[i]);
		Decode(board, possibilities.boards[i], possibilities.colors[i], possibilities.kings[i]);
		DisplayBoard(board);
	}
}

__device__ __host__ int CalculateScore(uint board, uint color, int flag)
{
	int whites = 0;
	int blacks = 0;
	if (flag) color = ~color;
	while (board > 0)
	{
		if (board & 1)
		{
			if (board & color & 1) whites++;
			else blacks++;
		}
		board >>= 1;
		color >>= 1;
	}
	return (whites << 5) | blacks;
}

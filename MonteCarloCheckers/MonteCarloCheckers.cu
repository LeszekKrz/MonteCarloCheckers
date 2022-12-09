
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <string>

typedef unsigned int uint;
typedef unsigned long long int ulong;


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
	bool kills[32];
	int wins[32];
};

__device__ __host__ RandomResult Random(RandomResult);

__device__ __host__ void DisplayBoard(char board[8][8]);

__device__ __host__ void Encode(char board[8][8], uint*, uint*, uint*);
__device__ __host__ void Decode(char board[8][8], uint, uint, uint);

__device__ __host__ void PrintBits(uint);

__device__ __host__ void Move(uint*, uint*, uint*, int, int);
__device__ __host__ void Remove(uint*, uint*, int);

__global__ void SimulateGame(RandomResult*, Possibilities*, int, bool);

void SimulateGameCPU(RandomResult*, Possibilities*, int,int);

__device__ __host__ int MakeMove(uint*, uint*, uint*, RandomResult*, bool);

__device__ __host__ bool MultipleHit(uint*, uint*, uint*, RandomResult*, int*);
__device__ __host__ int CalculateScore(uint, uint, int);

int FindPossibleMoves(uint, uint, uint, int, Possibilities*);
bool FindPossibleMultipleHit(uint, uint, uint, int, int, int, int*, Possibilities*, int, int);
void AddPossible(uint, uint, uint, int, int, int, int, Possibilities*, bool);
void DisplayPossibilities(Possibilities, int);

int ChessNotation(int);


int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	bool player = false;
	int playerTurn = -1;
	bool hint = false;
	bool wait = false;
	bool showMoves = true;
	bool onCpu = false;
	int thousands;
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
	for (int i = 0; i < 3; i++)
	{
		for (int j = 1 - i % 2; j < 8; j += 2)
		{
			checkersBoard[i][j] = 'Y';
			checkersBoard[7 - i][7 - j] = 'X';
		}
	}
	uint occupied = 0, color = 0, kings = 0;
	int i = 0;

	int choice = -1;

	printf("Na CPU czy GPU?\n");
	printf("1. CPU\n");
	printf("2. GPU\n");
	printf("Wpisz 1 lub 2: ");
	std::cin >> choice;
	if (choice == 1) onCpu = true;
	printf("\n");

	printf("Ile tysiecy symulacji na kazdy mozliwy ruch? ");
	std::cin >> thousands;
	printf("\n");

	printf("Wybierz:\n");
	printf("1. Od poczatku\n");
	printf("2. Wczytaj plansze\n");
	printf("Wpisz 1 lub 2: ");
	std::cin >> choice;
	printf("\n");
	
	if (choice == 2)
	{
		printf("Podaj zakodowany stan gry.\n");
		printf("Podaj stan zajetych pol: ");
		std::cin >> occupied;
		printf("Podaj stan kolorow: ");
		std::cin >> color;
		printf("Podaj stan damek: ");
		std::cin >> kings;

		printf("Wczytana plansza:\n");
		Decode(checkersBoard, occupied, color, kings);
		DisplayBoard(checkersBoard);
		printf("\n");
		printf("Czyja kolej:\n");
		printf("1. X\n");
		printf("2. Y\n");
		printf("Wpisz 1 lub 2: ");
		std::cin >> choice;
		if (choice == 1) i = 0;
		else
		{
			i = 1;
		}
		printf("\n");
	}

	printf("1. Gracz z komputerem\n");
	printf("2. Komputer z komputerem\n");
	printf("Wpisz 1 lub 2: ");
	std::cin >> choice;
	if (choice == 1) player = true;
	printf("\n");

	if (player)
	{
		printf("Czy chcesz podpowiedzi\n");
		printf("1. Tak\n");
		printf("2. Nie\n");
		printf("Wpisz 1 lub 2: ");
		std::cin >> choice;
		if (choice == 1) hint = true;
		printf("\n");

		printf("Kto zaczyna?\n");
		printf("1. Gracz\n");
		printf("2. Komputer\n");
		printf("Wpisz 1 lub 2: ");
		std::cin >> choice;
		if (choice == 1) playerTurn = i;
		else playerTurn = (i + 1) % 2;
		printf("\n");
	}
	
	printf("Czekac co ruch?\n");
	printf("1. Tak\n");
	printf("2. Nie\n");
	printf("Wpisz 1 lub 2: ");
	std::cin >> choice;
	if (choice == 1) wait = true;
	printf("\n");
	

	printf("Pokazywac decyzje komputera?\n");
	printf("1. Tak\n");
	printf("2. Nie\n");
	printf("Wpisz 1 lub 2: ");
	std::cin >> choice;
	if (choice == 2) showMoves = false;
	printf("\n");

	DisplayBoard(checkersBoard);
	printf("Nacisnij dowolny przycisk aby rozpoczac gre\n");
	getchar();
	getchar();
	
	int whites = 12, blacks = 12;
	int result;
	int moves = 0;
	int maxMoves = 0;



	RandomResult* d_random = 0;
	uint* d_board = 0;
	uint* d_colors = 0;
	uint* d_kings = 0;
	Possibilities* d_possibilities = 0;

	cudaError_t cudaStatus;

	int blocks;
	int kills;

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

	int maxI, max;
	int last = -1, lastI;
	
	printf("Poczatek symulacji...\n");
	Encode(checkersBoard, &occupied, &color, &kings);
	if (i) color = ~color;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	while(1)
	{
		int state = CalculateScore(occupied, color, i % 2);
		if (state != last)
		{
			lastI = i;
		}
		else
		{
			if (i - lastI > 30)
			{
				printf("Remis!");
				break;
			}
		}
		last = state;
		printf("Biali: %d Czarni %d\n", state >> 5, state & 31);
		if ((state >> 5) == 0 || (state & 31) == 0)
		{
			printf("Koniec gry!\n");
			break;
		}
		cudaEventRecord(start);
		blocks = FindPossibleMoves(occupied, color, kings, i%2, &possibilities);
		if (blocks == 0)
		{
			printf("Brak mozliwych ruchow. ");
			if (i%2) printf("Czarni przegrali\n");
			else printf("Biali przegrali\n");
			break;
		}


		kills = 0;
		for (int j = 0; j < blocks; j++) if (possibilities.kills[j]) kills++;


		printf("%d mozliwosci z czego %d bic\n", blocks, kills);

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

		if ((i % 2 == playerTurn && hint) || (i % 2 != playerTurn))
		{
			if (onCpu)
			{
				for (int k = 0; k < blocks; k++)
				{
					if (kills == 0 || (kills > 0 && possibilities.kills[i] == true))
					{
						for (int j = 0; j < thousands * 1000; j++)
						{
							SimulateGameCPU(&random, &possibilities, (i + 1) % 2, k);
						}
					}
				}
			}
			else
			{
				dim3 blocks3(blocks, 2*thousands, 1);
				SimulateGame << <blocks3, 500 >> > (d_random, d_possibilities, (i + 1) % 2, kills > 0);

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
			}

			maxI = -1;
			max = -1;
			for (int i = 0; i < blocks; i++)
			{
				if (possibilities.wins[i] > max && (!kills || possibilities.kills[i]))
				{
					maxI = i;
					max = possibilities.wins[i];
				}
			}
			if ((i%2 != playerTurn && showMoves) || (i%2 == playerTurn && hint))
			{
				for (int i = 0; i < blocks; i++) printf("Blok %d: %d\n", i + 1, possibilities.wins[i]);
				printf("Najlepsza mozliwosc %d\n", maxI + 1);
			}
		}
		if (i % 2 == playerTurn)
		{
			printf("Wybierz ruch: ");
			std::cin >> maxI;
			maxI--;
		}
		if (i % 2) printf("Y\n");
		else printf("X\n");
		occupied = possibilities.boards[maxI];
		color = possibilities.colors[maxI];
		kings = possibilities.kings[maxI];
		i++;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		miliseconds /= 1000;
		printf("Symulacja zakonczona po %f sekundach\n", miliseconds);
		if (i%2) Decode(checkersBoard, occupied, ~color, kings);
		else Decode(checkersBoard, occupied, color, kings);
		DisplayBoard(checkersBoard);
		if (wait) getchar();
	}
	printf("Gra zakonczyla sie po %d ruchach\n", i);


	printf("\n");

Error:
	cudaFree(d_random);
	cudaFree(d_possibilities);
	return 0;
}

__global__ void SimulateGame(RandomResult* o_random, Possibilities* o_possibilites, int turn, bool killing)
{
	if (killing && !(o_possibilites->kills[blockIdx.x]))
	{
		if (threadIdx.x == 0 && blockIdx.y == 0) o_possibilites->wins[blockIdx.x] = -1;
		return;
	}
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
	

	int result;
	for (int i = turn; i < 1000; i++)
	{
		result = MakeMove(board, colors, kings, random, i % 2);
		if (result == -1)
		{
			break;
		}
		else if (result > 0)
		{
			if (i % 2)
			{
				whites -= result;
				if (whites == 0)
				{
					break;
				}
			}
			else
			{
				blacks -= result;
				if (blacks == 0)
				{
					break;
				}
			}
		}
		*colors = ~(*colors);
	}
	if (turn) blacks = __syncthreads_count(whites);
	else blacks = __syncthreads_count(blacks);
	if (threadIdx.x == 0)
	{
		//printf("Nasi wygrali %d gier w bloku %d\n", blacks, blockIdx.x);
		//o_possibilites->wins[blockIdx.x] += blacks;
		atomicAdd(&(o_possibilites->wins[blockIdx.x]), blacks);
	}
}

void SimulateGameCPU(RandomResult* o_random, Possibilities* o_possibilities, int turn, int move)
{
	uint m_board = o_possibilities->boards[move];
	uint m_colors = o_possibilities->colors[move];
	uint m_kings = o_possibilities->kings[move];
	uint* board = &m_board;
	uint* colors = &m_colors;
	uint* kings = &m_kings;
	RandomResult* random = o_random;
	int whites = CalculateScore(*board, *colors, turn);
	int blacks = whites & 31;
	whites >>= 5;

	int result;
	char checkersBoard[8][8];
	for (int i = turn; i < 1000; i++)
	{
		result = MakeMove(board, colors, kings, random, i % 2);
		/*if (i % 2) Decode(checkersBoard, *board, ~(*colors), *kings);
		else Decode(checkersBoard, *board, *colors, *kings);
		DisplayBoard(checkersBoard);*/
		if (result == -1)
		{
			break;
		}
		else if (result > 0)
		{
			if (i % 2)
			{
				whites -= result;
				if (whites == 0)
				{
					break;
				}
			}
			else
			{
				blacks -= result;
				if (blacks == 0)
				{
					break;
				}
			}
		}
		*colors = ~(*colors);
	}
	if (turn)
	{
		o_possibilities->wins[move] += whites > 0 ? 1 : 0;
	}
	else
	{
		o_possibilities->wins[move] += blacks > 0 ? 1 : 0;
	}
}

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


void Add(uint* board, uint* kings, int n, bool king)
{
	(*board) |= 1 << n;
	if (king) (*kings) |= 1 << n;
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
								killing = true;
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
							if (!((*occupied) & (1 << n - 7))) // podwojny skos mozliwy
							{
								count++;
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
								killing = true;
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
							if (!((*occupied) & (1 << n + 7))) // podwojny skos mozliwy
							{
								count++;
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
								killing = true;
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
							if (!((*occupied) & (1 << n + 9))) // podwojny skos mozliwy
							{
								count++;
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
								killing = true;
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


	if (count == 0)
	{
		return -1;
	}
	t_occupied = (*occupied) & (*color);
	int moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
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
									if (count == moves)
									{
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
										Move(occupied, color, kings, n, t_n - n - 4 - k);
										Remove(occupied, kings, t_n);
										count = 1;
										n = t_n - 4 - k;
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
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * (-9) + (moves & 1) * (-4 - k));
									return 0;
								}
							}
							else if (!killing && !flag)
							{
								count++;
								if (count == moves)
								{
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
									if (count == moves)
									{
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
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * (-7) + (moves & 1) * (-3 - k));
									return 0;
								}
								k = (n >> 2) & 1;
							}
							else if (!killing && !flag)
							{
								count++;
								if (count == moves)
								{
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
									if (count == moves)
									{
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
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * 7 + (moves & 1) * (4 - k));
									return 0;
								}
							}
							else if (!killing && flag)
							{
								count++;
								if (count == moves)
								{
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
									if (count == moves)
									{
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
									k = (n >> 2) & 1;
									Move(occupied, color, kings, n, (moves >> 1) * 9 + (moves & 1) * (5 - k));
									return 0;
								}
							}
							if (!killing && flag)
							{
								count++;
								if (count == moves)
								{
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
			return false;
		}
	}
}

int FindPossibleMoves(uint occupied, uint color, uint kings, int flag, Possibilities* possibilities)
{
	int possible = 0;
	bool kills = false;
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
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, n, -9, n - 4 - k, &possible, possibilities, n, flag))
								{
									if ((n - 9 >> 2) == flag * 7) kings |= 1 << (n - 9);
									AddPossible(occupied, color, kings, n, -9, n - 4 - k, possible, possibilities, ((n - 9) >> 2) == flag * 7);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || !flag) && !((occupied) & (1 << n - 4 - k))) // wolne lewo gora
					{
						AddPossible(occupied, color, kings, n, -4 - k, -1, possible, possibilities, ((n - 4 - k) >> 2) == flag * 7);
						possible++;
						if (kings & (1 << n))
						{
							t_n = n - 4 - k;
							k ^= 1;
							if ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								t_n += -4 - k;
								k ^= 1;
							}
							while ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								possible++;
								t_n += -4 - k;
								k ^= 1;
							}
							if ((t_n >> 2) == flag * 7 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							else if (((t_n & 3) + 1 - k == 0) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								possible++;
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n - 4 - k)))
							{
								Move(&occupied, &color, &kings, n, t_n + 4 + 1 - k);
								bool rememberKing = kings & t_n;
								Remove(&occupied, &kings, t_n);
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, t_n + 4 + 1 - k, -4 - k, t_n, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, t_n - 4 - k - n, t_n, possible, possibilities, ((t_n - 4 - k) >> 2) == flag * 7);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n + 4 + 1 - k, n);
								Add(&occupied, &kings, t_n, rememberKing);
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
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, n, -7, n - 3 - k, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, -7, n - 3 - k, possible, possibilities, ((n - 7) >> 2) == flag * 7);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || !flag) && !((occupied) & (1 << n - 3 - k))) // wolne prawo gora
					{
						AddPossible(occupied, color, kings, n, -3 - k, -1, possible, possibilities, ((n - 3 - k) >> 2) == flag * 7);
						possible++;
						if (kings & (1 << n))
						{
							t_n = n - 3 - k;
							k ^= 1;
							if ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								t_n += -3 - k;
								k ^= 1;
							}
							while ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								t_n += -3 - k;
								k ^= 1;
								possible++;
							}
							if ((t_n >> 2) == flag * 7 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							else if ((t_n & 3) - k == 3 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								possible++;
							}
							if ((t_n >> 2) > 0 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n - 3 - k)))
							{
								Move(&occupied, &color, &kings, n, t_n + 3 + 1 - k);
								bool rememberKing = kings & t_n;
								Remove(&occupied, &kings, t_n);
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, t_n + 3 + 1 - k, -3 - k, t_n, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, t_n - 3 - k - n, t_n, possible, possibilities, ((t_n - 3 - k) >> 2) == flag * 7);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n + 3 + 1 - k, n);
								Add(&occupied, &kings, t_n, rememberKing);
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
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, n, 7, n + 4 - k, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, 7, n + 4 - k, possible, possibilities, ((n + 7) >> 2) == flag * 7);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || flag) && !((occupied) & (1 << n + 4 - k)))// wolne lewo dol
					{
						AddPossible(occupied, color, kings, n, 4 - k, -1, possible, possibilities, ((n + 4 - k) >> 2) == flag * 7);
						possible++;
						if (kings & (1 << n))
						{
							t_n = n + 4 - k;
							k ^= 1;
							if ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								t_n += 4 - k;
								k ^= 1;
							}
							while ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								t_n += 4 - k;
								k ^= 1;
								possible++;
							}
							if ((t_n >> 2) == flag * 7 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							else if ((t_n & 3) + 1 - k == 0 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) + 1 - k) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n + 4 - k)))
							{
								Move(&occupied, &color, &kings, n, t_n - 4 + 1 - k);
								bool rememberKing = kings & t_n;
								Remove(&occupied, &kings, t_n);
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, t_n - 4 + 1 - k, 4 - k, t_n, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, t_n + 4 - k - n, t_n, possible, possibilities, ((t_n + 4 - k) >> 2) == flag * 7);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n - 4 + 1 - k, n);
								Add(&occupied, &kings, t_n, rememberKing);
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
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, n, 9, n + 5 - k, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, 9, n + 5 - k, possible, possibilities, ((n + 9) >> 2) == flag * 7);
									possible++;
								}
							}
						}
					}
					else if ((kings & (1 << n) || flag) && !((occupied) & (1 << n + 5 - k))) // wolne prawo dol
					{
						AddPossible(occupied, color, kings, n, 5 - k, -1, possible, possibilities, ((n + 5 - k) >> 2) == flag * 7);
						possible++;
						if (kings & (1 << n))
						{
							t_n = n + 5 - k;
							k ^= 1;
							if ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								t_n += 5 - k;
								k ^= 1;
							}
							while ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, false);
								t_n += 5 - k;
								k ^= 1;
								possible++;
							}
							if ((t_n >> 2) == flag * 7 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							else if ((t_n & 3) - k == 3 && !(occupied & (1 << t_n)))
							{
								AddPossible(occupied, color, kings, n, t_n - n, -1, possible, possibilities, true);
								possible++;
							}
							if ((t_n >> 2) < 7 && ((t_n & 3) - k < 3) && (enemies & (1 << t_n)) && !(occupied & (1 << t_n + 5 - k)))
							{
								Move(&occupied, &color, &kings, n, t_n - 5 + 1 - k);
								bool rememberKing = kings & t_n;
								Remove(&occupied, &kings, t_n);
								kills = true;
								if (FindPossibleMultipleHit(occupied, color, kings, t_n - 5 + 1 - k, 5 - k, t_n, &possible, possibilities, n, flag))
								{
									AddPossible(occupied, color, kings, n, t_n + 5 - k - n, t_n, possible, possibilities, ((t_n + 5 - k) >> 2) == flag * 7);
									possible++;
								}
								Move(&occupied, &color, &kings, t_n - 5 + 1 - k, n);
								Add(&occupied, &kings, t_n, rememberKing);
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

bool FindPossibleMultipleHit(uint occupied, uint color, uint kings, int n, int move, int hit, int* possible, Possibilities* possibilities, int origin, int flag)
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
			if ((enemies & (1 << n - 4 - k)) && !(occupied & (1 << n - 9))) // raz zajete dwa wolne
			{
				end = false;
				if (FindPossibleMultipleHit(occupied, color, kings, n, -9, n - 4 - k, possible, possibilities, origin, flag))
				{
					AddPossible(occupied, color, kings, origin, n - 9 - origin, n - 4 - k, *possible, possibilities, ((n - 9) >> 2) == flag * 7);
					(*possible)++;
				}
			}
		}
		if ((n & 3) < 3) // mozna dwa razy w prawo
		{
			if ((enemies & (1 << n - 3 - k)) && !((occupied) & (1 << n - 7))) // raz zajete dwa wolne
			{
				end = false;
				if (FindPossibleMultipleHit(occupied, color, kings, n, -7, n - 3 - k, possible, possibilities, n, flag))
				{
					AddPossible(occupied, color, kings, origin, n - 7 - origin, n - 3 - k, *possible, possibilities, ((n - 7) >> 2) == flag * 7);
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
				if (FindPossibleMultipleHit(occupied, color, kings, n, 7, n + 4 - k, possible, possibilities, n, flag))
				{
					AddPossible(occupied, color, kings, origin, n + 7 - origin, n + 4 - k, *possible, possibilities, ((n + 7) >> 2) == flag * 7);
					(*possible)++;
				}
			}
		}
		if ((n & 3) < 3) // mozna dwa razy w prawo
		{
			if ((enemies & (1 << n + 5 - k)) && !((occupied) & (1 << n + 9))) // raz zajete dwa wolne
			{
				end = false;
				if (FindPossibleMultipleHit(occupied, color, kings, n, 9, n + 5 - k, possible, possibilities, n, flag))
				{
					AddPossible(occupied, color, kings, origin, n + 9 - origin, n + 5 - k, *possible, possibilities, ((n + 9) >> 2) == flag * 7);
					(* possible)++;
				}
			}
		}
	}
	return end;
}

void AddPossible(uint occupied, uint color, uint kings, int n, int move, int hit, int possibleIndex, Possibilities* possibilities, bool coronation)
{
	if (move != 0) Move(&occupied, &color, &kings, n, move);
	if (hit >= 0) Remove(&occupied, &kings, hit);
	if (coronation) kings |= 1 << (n + move);
	possibilities->origins[possibleIndex] = n;
	possibilities->boards[possibleIndex] = occupied;
	possibilities->colors[possibleIndex] = ~color;
	possibilities->kings[possibleIndex] = kings;
	possibilities->kills[possibleIndex] = hit >= 0;
	possibilities->wins[possibleIndex] = 0;
	int from = ChessNotation(n);
	int to = ChessNotation(n + move);
	if (hit >= 0) printf("%d.Bicie  %c%d -> %c%d\n", possibleIndex + 1, 'A' + (from & 7), from >> 3, 'A' + (to & 7), to >> 3);
	else printf("%d.Ruch %c%d -> %c%d\n", possibleIndex + 1, 'A' + (from & 7), from >> 3, 'A' + (to & 7), to >> 3);
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


int ChessNotation(int n)
{
	return ((8 - (n >> 2)) << 3) + ((n & 3) << 1) + 1 - ((n >> 2) & 1);
}
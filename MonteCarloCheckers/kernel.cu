
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef unsigned int uint;
typedef unsigned long long int ulong;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

uint TausStep(uint, int, int, int, uint);
uint LCGStep(uint z, uint A, uint C);

typedef struct RandomResult
{
	int x;
	int y;
	int z;
	int w;
	float value;
} RandomResult;

RandomResult Random(RandomResult);

void DisplayBoard(char board[8][8]);

void Encode(char board[8][8], uint*, uint*, uint*);
void Decode(char board[8][8], uint, uint, uint);

void PrintBits(uint);

void Move(uint*, uint*, uint*, int, int);
void Remove(uint*, uint*, int);

int MakeMove(uint*, uint*, uint*, RandomResult*, bool);

bool MultipleHit(uint*, uint*, uint*, RandomResult*, int*);


int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };


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
	occupied |= 1 << 18;
	occupied |= 1 << 17;
	color |= 1 << 18;
	kings |= 1 << 18;
	Decode(checkersBoard, occupied, color, kings);
	DisplayBoard(checkersBoard);
	MakeMove(&occupied, &color, &kings, &random, 0);
	Decode(checkersBoard, occupied, color, kings);
	DisplayBoard(checkersBoard);

	/*Encode(checkersBoard, &occupied, &color, &kings);
	PrintBits(occupied);
	PrintBits(color);
	PrintBits(kings);
	int blacks = 12, whites = 12;
	int result;
	for (int i = 0; i < 100; i++)
	{
		result = MakeMove(&occupied, &color, &kings, &random, i % 2);
		if (result == -1)
		{
			printf("Koniec gry!\n");
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
			DisplayBoard(checkersBoard);
			break;
		}
		else if (result > 0)
		{
			if (i % 2)
			{
				whites -= result;
				if (whites == 0)
				{
					printf("Biali przegrali!\n");
					Decode(checkersBoard, occupied, color, kings);
					DisplayBoard(checkersBoard);
					break;
				}
			}
			else
			{
				blacks -= result;
				if (blacks == 0)
				{
					printf("Czarni przegrali\n");
					Decode(checkersBoard, occupied, color, kings);
					DisplayBoard(checkersBoard);
					break;
				}
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
		DisplayBoard(checkersBoard);
		color = ~color;
	}*/
	//MakeMove(&occupied, &color, &random);
	//Decode(checkersBoard, occupied, color);
	//DisplayBoard(checkersBoard);
	//PrintBits(occupied);
	//PrintBits(color);
	//Move(&occupied, &color, 11, 4);
	//Decode(checkersBoard, occupied, color);
	//DisplayBoard(checkersBoard);
	//PrintBits(occupied);
	//PrintBits(UINT_MAX);
	printf("\n");

	//  printf("%d %d %d %d\n", random.x, random.y, random.z, random.w);
	//  for (int i = 0; i < 20; i++)
	//  {
		  //random = Random(random);
		  ////printf("%f\n",(int)((random.value + 0.5) / 0.1));
	//      //printf("%f %f %d\n", (random.value + 0.5), (random.value + 0.5) / (1.0/12), (int)((random.value + 0.5) / (1.0/12)));
	//  }




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

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);
}

RandomResult Random(RandomResult random)
{
	RandomResult result;
	result.x = TausStep(random.x, 13, 19, 12, 4294967294);
	result.y = TausStep(random.y, 2, 25, 4, 4294967288);
	result.z = TausStep(random.z, 3, 11, 17, 4294967280);
	result.w = LCGStep(random.w, 1664525, 1013904223);

	result.value = 2.3283064365387e-10 * (result.x ^ result.y ^ result.z ^ result.w); // -0.5 0.5

	return result;
}

void DisplayBoard(char board[8][8])
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

void Encode(char board[8][8], uint* occupied, uint* color, uint* kings)
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

void Decode(char board[8][8], uint occupied, uint color, uint kings)
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

//Coordinates Convert(int n)
//{
//    return ((n/4) % 2 == 0)?()
//}


void Move(uint* board, uint* color, uint* kings, int n, int d)
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

void Remove(uint* board, uint* kings, int n)
{
	(*board) &= ~(1 << n);
	(*kings) &= ~(1 << n);
}

int MakeMove(uint* occupied, uint* color, uint* kings, RandomResult* random, bool flag) // zakladam ze rusza sie tylko kolor oznaczony 1
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
								printf("morderczy krol\n");
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
								printf("Morderczy krol\n");
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
								printf("Morderczy krol\n");
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
								printf("Morderczy krol\n");
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



	if (count == 0)
	{
		printf("Nie można wykonać ruchu!\n");
		return -1;
	}
	if (killing) printf("Musze bic!\n");
	t_occupied = (*occupied) & (*color);

	//int moves = (int)((random->value + 0.5) / (1.0 / count)) + 1;
	int moves = 1;
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
										printf("Bicie! %d -> %d\n", n, -9);
										Remove(occupied,kings, n - 4 - k);
										Move(occupied, color, kings, n, -9);
										count = 1;
										n -= 9;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
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
										printf("Krol atakuje! %d -> %d\n", n, t_n - n - 4 - k);
										Move(occupied, color, kings, n, t_n - n - 4 - k);
										Remove(occupied, kings, t_n);
										PrintBits(*occupied);
										PrintBits(*color);
										return 1;
									}
									k = (n >> 2) & 1;
								}
							}
							if (!killing && !flag)
							{
								count++;
								//printf("Znaleziono %d ruch %d -> %d lg\n", count, n, -4 - k);
								if (count == moves)
								{
									//printf("Wykonany ruch!\n");
									if (*kings & (1 << n))
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
										printf("Wykrylem %d ruchow i wykonuje %d ruch\n", count, moves);
										k = (n >> 2) & 1;
										Move(occupied, color, kings, n, (moves >> 1) * (-9) + (moves & 1) * (-4 - k));
										return 0;
									}
									Move(occupied, color, kings, n, -4 - k);
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
										printf("Bicie! %d -> %d\n", n, -7);
										Remove(occupied, kings,n - 3 - k);
										Move(occupied, color, kings, n, -7);
										count = 1;
										n -= 7;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
							}
						}
						else if (!killing && !flag && !((*occupied) & (1 << n - 3 - k))) // wolne prawo gora
						{
							count++;
							//printf("Znaleziono %d ruch %d -> %d pg\n", count, n, -3 -k);
							if (count == moves)
							{
								//printf("Wykonany ruch!\n");
								Move(occupied, color, kings, n, -3 - k);
								return 0;
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
										printf("Bicie! %d -> %d\n", n, 7);
										Remove(occupied, kings,n + 4 - k);
										Move(occupied, color, kings, n, 7);
										count = 1;
										n += 7;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
							}
						}
						else if (!killing && flag && !((*occupied) & (1 << n + 4 - k)))// wolne lewo dol
						{
							count++;
							//printf("Znaleziono %d ruch %d -> %d ld\n", count, n, 4 - k);
							if (count == moves)
							{
								//printf("Wykonany ruch!\n");
								Move(occupied, color, kings, n, 4 - k);
								return 0;
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
									//printf("Znaleziono %d ruch %d -> %d pdpd\n", count, n, 9);
									if (count == moves)
									{
										//printf("Wykonany ruch!\n");
										printf("Bicie! %d -> %d\n", n, 9);
										Remove(occupied, kings,n + 5 - k);
										Move(occupied, color, kings, n, 9);
										count = 1;
										n += 9;
										while (MultipleHit(occupied, color, kings, random, &n)) count++;
										return count;
									}
								}
							}
						}
						else if (!killing && flag && !((*occupied) & (1 << n + 5 - k))) // wolne
						{
							count++;
							//printf("Znaleziono %d ruch %d -> %d pd\n", count, n, 5 -k);
							if (count == moves)
							{
								//printf("Wykonany ruch!\n");
								Move(occupied, color, kings, n, 5 - k);
								return 0;
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

bool MultipleHit(uint* occupied, uint* color, uint* kings, RandomResult* random, int* n)
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
						printf("Kontynuuje bicie! %d -> %d\n", *n, -9);
						Move(occupied, color, kings, *n, -9);
						Remove(occupied, kings,*n - 4 - k);
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
						printf("Kontynuuje bicie! %d -> %d\n", *n, -7);
						Move(occupied, color, kings, *n, -7);
						Remove(occupied, kings,*n - 3 - k);
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
						printf("Kontynuuje bicie! %d -> %d\n", *n, 7);
						Move(occupied, color, kings, *n, 7);
						Remove(occupied, kings,*n + 4 - k);
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
						printf("Kontynuuje bicie! %d -> %d\n", *n, 9);
						Move(occupied, color, kings, *n, 9);
						Remove(occupied, kings,*n + 5 - k);
						*n = *n + 9;
						return true;
					}
				}
			}
		}

		if (count == 0) // koniec bicia
		{
			printf("Koniec bicia!\n");
			return false;
		}
	}
}

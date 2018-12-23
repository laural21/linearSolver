/* Lab 1 - Laura Lehoczki
 *
 * This parallel program finds the unknowns in a set of linear equations
 * based on some original values and an absolute relative error.
 *
 * To compile: mpicc -g -Wall -std=c99 -o gs gs.c
 * To run: mpiexec -n <num processes: 1/2/10/20/40> ./gs
 *
 * Algorithm:
 * 1. Read in all data
 * 2. Compute how many unknowns each process has to evaluate
 * 3. Each process calculates new value for its unknowns
 * 4. MPI_Allgather collects new all new x's
 * 5. Error rates are checked for all new x values
 * 6. If not all errors are below the absolute relative error, each process calculates a new value
 *    for its x's.
 * 7. If all errors are below absolute relative error
 *
 *
 * else - if i'm not core 0:
 *  receive coefficients, x's, constants, err
 *  calculate x[my_rank]
 *  calculate relative error
 *  send error and new x to core 0
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */

/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
float* calc_unknown(float *x, int local_lower, int local_upper, int local_n);
int check_error(float *x_old, float *x_new, int local_lower, int local_upper);  /* Check if relative error for equation is smaller than required */
/********************************/

/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
    Calculate the unknowns for your rank
*/
float* calc_unknown(float *x, int local_lower, int local_upper, int local_n){
    float *x_new_local = (float *)malloc(local_n* sizeof(float *));
    float sum = 0;
    int i, j;
    for (i = local_lower; i < local_upper; i++){
        for (j = 0; j < num; j++){
            if(i != j){
                sum += a[i][j] * x[j];
            }
        }
        x_new_local[i] = (b[i]-sum)/(a[i][i]);
        printf("%f ", x_new_local[i]);
    }
    return x_new_local;
}

/*
    This function checks if the new values of the unknowns are all below the absolute relative
    error. If some are not, calc_unknown will have to be repeated.
    Note that e is not multiplied by 100, because err was not either.
 */

int check_error(float *x_old, float *x_new, int local_lower, int local_upper){
    float e;
    int i;
    for (i = local_lower; i < local_upper; i++){
        e = abs((x_new[i] - x_old[i])/x_new[i]);
        if (abs(e) > err){
            return 0;
        }
    }
    return 1;
}

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}

/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
  fclose(fp);
}

/************************************************************/


int main(int argc, char *argv[])
{
 int i;
 int nit = 0; /* number of iterations */
 FILE * fp;
 char output[100] ="";
  
 if( argc != 2)
 {
   printf("Usage: ./gsref filename\n");
   exit(1);
 }

    /* Check for convergence condition */
    /* This function will exit the program if the coffeicient will never converge to
     * the needed absolute error.
     * This is not expected to happen for this programming assignment.
     */
    //check_matrix();
    get_input(argv[1]);

 float *x_new = (float *)malloc(num* sizeof(float *));
 int my_rank, comm_sz;

 /* Set up MPI */
 MPI_Init(NULL, NULL);
 MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //Find out my rank

 /* Find out how many processes are being used and give each a number of equations*/
 MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int local_n = num/comm_sz;
    int local_lower = my_rank*local_n;
    int local_upper = local_lower+local_n;
    float *x_new_local = (float *)malloc(local_n* sizeof(float*));
    printf("pointer to new local: %f", x_new_local);
    /* Check if variables are correctly allocated */
    //printf("my rank: %d, comm_size: %d, local_n: %d, local_lower: %d, local_upper: %d", my_rank, comm_sz, local_n, local_lower, local_upper);


/*
 void Build_mpi_type(float** a_p, float* b_p, float* x_p, float* e, MPI_Datatype* sent_data){
    int a_blocklengths[4] = {num, 1, 1, 1};
    MPI_Datatype a_types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint a_addr, b_addr, x_addr, e_addr;
    MPI_Aint a_displacements[3] = {0};
    MPI_Get_address(a_p, &a_addr);
    MPI_Get_address(b_p, &b_addr);
    MPI_Get_address(x_p, &x_addr);
    MPI_Get_address(e, &e_addr);
    a_displacements[1] = b_addr - a_addr;
    a_displacements[2] = x_addr - b_addr;
    a_displacements[3] = e_addr - x_addr;
    MPI_Type_create_struct(4, a_blocklengths, a_displacements, a_types, sent_data);
    MPI_Type_commit(sent_data);
 }

    MPI_Datatype *sent_data;
    Build_mpi_type(a, b, x, e, sent_data);

    // Scatter arrays to all processes
    MPI_Scatter(a, local_n, sent_data, a, local_n, sent_data, 0, MPI_COMM_WORLD);
*/

 /*
    Calculate new values for unknowns.
    Collect all calculated values and check if they are all below the abs. rel. error.
    MPI_Allgather is a blocking call, so it acts as a natural barrier.
 */
    x_new_local = calc_unknown(x, local_lower, local_upper, local_n);
    nit++;
    MPI_Allgather(x_new_local, local_n, MPI_FLOAT, x_new, num, MPI_FLOAT, MPI_COMM_WORLD);
    //printf("%f, %f", x_new_local, x_new);

    while(!check_error(x, x_new, local_lower, local_upper)){
        nit++;
        x = x_new;
        x_new_local = calc_unknown(x, local_lower, local_upper, local_n);
        MPI_Allgather(x_new_local, local_n, MPI_FLOAT, x_new, num, MPI_FLOAT, MPI_COMM_WORLD);
    }

    free(x_new_local);

    // Process 0 writes results to file and prints number of iterations
    if(my_rank == 0){

     sprintf(output,"%d.sol",num);
     fp = fopen(output,"w");
     if(!fp)
     {
         printf("Cannot create the file %s\n", output);
         exit(1);
     }

     for( i = 0; i < num; i++)
         fprintf(fp,"%f\n",x[i]);

     printf("total number of iterations: %d\n", nit/num);

     fclose(fp);
 }

    // Free memory
    for (int i = 0; i < num; ++i)
        free(a[i]);
    free(a);
    free(x);
    free(b);
    free(x_new);

 exit(0);
 MPI_Finalize();
 return 0;
}
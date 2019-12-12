#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

int
main (int argc, char *argv[]) {
  int numX, numX2;
  FILE* fid = fopen(argv[1], "r");
  float *Qr1, *Qr2, *Qi1, *Qi2;

  if(argc != 3) {
    printf("Program takes 2 files as inputs to comparei. You have to few or to many inputs.\n");
    return -1;
  }

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
  fread(&numX, sizeof (int), 1, fid);
  
  printf("numX %d\n", numX);

  Qr1 = (float *) memalign(16, numX * sizeof (float));
  Qi1 = (float *) memalign(16, numX * sizeof (float));

  fread(Qr1, sizeof (float), numX, fid);
  fread(Qi1, sizeof (float), numX, fid);
  fclose (fid);

  fid = fopen(argv[2], "r");
  if (fid == NULL) {
    fprintf(stderr, "Cannot open input file\n");
    exit(-1);
  }
  fread(&numX2, sizeof (int), 1, fid);
  
  Qr2 = (float *) memalign(16, numX2 * sizeof (float));
  Qi2 = (float *) memalign(16, numX2 * sizeof (float));

  fread(Qr2, sizeof (float), numX2, fid);
  fread(Qi2, sizeof (float), numX2, fid);
  fclose (fid);

  printf("numX2 %d\n", numX2);

  if(numX != numX2) {
    printf("You do not have the same number of values to compare\n");
    return -1;
  }

  float diffCheck = .00006;

  float max = 0.00;
  float diff = 0.00;
  for(int i = 0; i < numX; ++i) {
    diff = Qr1[i] - Qr2[i];
    if(diff < 0)
      diff *= -1;
    if(diff > max)
      max = diff;
    printf("QR,diff %f\n", diff);
    if(diff > diffCheck) {
      printf("%f, %f, %d\n", Qr1[i], Qr2[i], i);
      printf("diff too great found %f\n", diff);
      return 2;
    }
    diff = Qi1[i] - Qi2[i];
    if(diff < 0)
      diff *= -1;
    if(diff > max)
      max = diff;
    printf("Qi,diff %f\n", diff);
    if(diff > diffCheck) {
      printf("%f, %f, %d\n", Qr1[i], Qr2[i], i);
      printf("diff too great found %f\n", diff);
      return 3;
    }
  }
  printf("max diff found %f\n", max);

}
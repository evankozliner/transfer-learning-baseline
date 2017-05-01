#include "mex.h"
#include <image.h>

#define EPS ( 1e-4 )
#define IS_ZERO_EPS( X ) ( fabs ( ( X ) ) < EPS )

#define BORDER_PATH "borders/"
#define VERBOSE 1
#define WRITE 1
#define NUM_METHODS 4

typedef struct
{
 int red, green, blue;
} Color;

int isnan ( double );

static int
method_huang ( const Image * img )
{
 SET_FUNC_NAME ( "method_huang" );
 int ih, it;
 int threshold;
 int first_bin;
 int last_bin;
 int sum_pix;
 int num_pix;
 int *data;			/* histogram data */
 double delta;
 double term;
 double tot_ent;		/* fuzzy entropy */
 double min_ent;		/* min entropy */
 double mu_x;
 double *mu_0, *mu_1;
 Histo *histo;			/* histogram */

 if ( !is_gray_img ( img ) )
  {
   ERROR_RET ( "Not a grayscale image !", INT_MIN );
  }

 /* Calculate the histogram */
 histo = create_histo ( img );

 /* Note that this is NOT the normalized histogram */
 data = get_histo_data ( histo );

 /* Determine the first non-zero bin */
 first_bin = 0;
 for ( ih = 0; ih < NUM_GRAY; ih++ )
  {
   if ( data[ih] != 0 )
    {
     first_bin = ih;
     break;
    }
  }

 /* Determine the last non-zero bin */
 last_bin = MAX_GRAY;
 for ( ih = MAX_GRAY; ih >= first_bin; ih-- )
  {
   if ( data[ih] != 0 )
    {
     last_bin = ih;
     break;
    }
  }

 /* 
    Equation (4) in Ref. 1
    C = g_max - g_min
    This is the term ( 1 / C ) 
  */
 term = 1.0 / ( double ) ( last_bin - first_bin );

 /* Mean for class #0 -- Equation (2) in Ref. 1 */
 mu_0 = ( double * ) calloc ( NUM_GRAY, sizeof ( double ) );

 sum_pix = num_pix = 0;
 for ( ih = first_bin; ih <= last_bin; ih++ )
  {
   sum_pix += ih * data[ih];
   num_pix += data[ih];

   /* NUM_PIX cannot be zero ! */
   mu_0[ih] = sum_pix / ( double ) num_pix;
  }

 /* Mean for class #1 -- Equation (3) in Ref. 1 */
 mu_1 = ( double * ) calloc ( NUM_GRAY, sizeof ( double ) );

 sum_pix = num_pix = 0;
 for ( ih = last_bin; ih > first_bin; ih-- )
  {
   sum_pix += ih * data[ih];
   num_pix += data[ih];

   /* NUM_PIX cannot be zero ! */
   mu_1[ih - 1] = sum_pix / ( double ) num_pix;
  }

 #define ENT( X ) ( -( X ) * log ( ( X ) ) )

 /* Determine the threshold that minimizes the fuzzy entropy */
 threshold = INT_MIN;
 min_ent = DBL_MAX;
 for ( it = first_bin; it <= last_bin; it++ )
  {
   tot_ent = 0.0;

   /* When IH < FIRST_BIN or IH > LAST_BIN => data[IH] = 0 */
   for ( ih = first_bin; ih <= last_bin; ih++ )
    {
     delta = fabs ( ih - ( ih <= it ? mu_0[it] : mu_1[it] ) );

     /* When DELTA = 0 => MU_X = 1 => ENT ( 1.0 - MU_X ) undefined */
     if ( !IS_ZERO_EPS ( delta ) )
      {
       /* 
        * Membership of gray level IH -- Equation (4) in Ref. 1.
        * This value is always in [0.5,1].
        */
       mu_x = 1.0 / ( 1.0 + term * delta );

       /* Equation (6) & (8) in Ref. 1 */
       tot_ent += data[ih] * ( ENT ( mu_x ) + ENT ( 1.0 - mu_x ) );
      }
    }

   /* No need to divide by NUM_ROWS * NUM_COLS * LOG(2) ! */

   if ( tot_ent < min_ent )
    {
     min_ent = tot_ent;
     threshold = it;
    }
  }

 #undef ENT

 free_histo ( histo );
 free ( mu_0 );
 free ( mu_1 );

 return threshold;
}

static int
method_kapur ( const Image * img )
{
 SET_FUNC_NAME ( "method_kapur" );
 int ih, it;
 int threshold;
 int first_bin;			/* see below */
 int last_bin;			/* see below */
 double term;
 double tot_ent;		/* total entropy */
 double max_ent;		/* max entropy */
 double ent_back;		/* entropy of the background pixels at a given threshold */
 double ent_obj;		/* entropy of the object pixels at a given threshold */
 double *data;			/* normalized histogram data */
 double *P1;			/* cumulative normalized histogram */
 double *P2;			/* see below */
 Histo *histo;		        /* histogram */
 Histo *norm_histo;		/* normalized histogram */

 if ( !is_gray_img ( img ) )
  {
   ERROR_RET ( "Not a grayscale image !", INT_MIN );
  }

 /* Calculate the histogram */
 histo = create_histo ( img );

 /* Calculate the normalized histogram */
 norm_histo = normalize_histo ( histo );

 data = get_histo_data ( norm_histo );

 /* Calculate the cumulative normalized histogram */
 P1 = accumulate_histo ( norm_histo );

 P2 = ( double * ) malloc ( NUM_GRAY * sizeof ( double ) );

 for ( ih = 0; ih < NUM_GRAY; ih++ )
  {
   P2[ih] = 1.0 - P1[ih];
  }

 /* 
    Determine the first non-zero 
    bin starting from the first bin 
  */
 first_bin = 0;
 for ( ih = 0; ih < NUM_GRAY; ih++ )
  {
   if ( !IS_ZERO_EPS ( P1[ih] ) )
    {
     first_bin = ih;
     break;
    }
  }

 /* 
    Determine the first non-one bin 
    starting from the last bin
  */
 last_bin = MAX_GRAY;
 for ( ih = MAX_GRAY; ih >= first_bin; ih-- )
  {
   if ( !IS_ZERO_EPS ( P2[ih] ) )
    {
     last_bin = ih;
     break;
    }
  }

 #define ENT( X ) ( -( X ) * log ( ( X ) ) )
 
 /* 
    Calculate the total entropy each gray-level
    and find the threshold that maximizes it 
  */
 threshold = INT_MIN;
 max_ent = DBL_MIN;
 for ( it = first_bin; it <= last_bin; it++ )
  {
   /* Entropy of the background pixels */
   ent_back = 0.0;
   term = 1.0 / P1[it];
   /* When IH < FIRST_BIN => DATA[IH] = 0 */
   for ( ih = first_bin; ih <= it; ih++ )
    {
     if ( !IS_ZERO_EPS ( data[ih] ) )
      {
       ent_back += ENT ( data[ih] * term );
      }
    }

   /* Entropy of the object pixels */
   ent_obj = 0.0;
   term = 1.0 / P2[it];
   for ( ih = it + 1; ih < NUM_GRAY; ih++ )
    {
     if ( !IS_ZERO_EPS ( data[ih] ) )
      {
       ent_obj += ENT ( data[ih] * term );
      }
    }

   /* Total entropy */
   tot_ent = ent_back + ent_obj;

   /* printf ( "T = %d ; ent_back = %f [P1 = %f]; ent_obj = %f [P2 = %f]; ent_total = %f\n", it, ent_back, P1[it], ent_obj, P2[it], tot_ent ); */

   if ( max_ent < tot_ent )
    {
     max_ent = tot_ent;
     threshold = it;
    }
  }

 #undef ENT

 free_histo ( histo );
 free_histo ( norm_histo );
 free ( P1 );
 free ( P2 );

 return threshold;
}

/* 
 * Iterative Variant of Kittler & Illingworth's Minimum Error Thresholding Method. 
 * Notation based on "An Analysis of Histogram-Based Thresholding Algorithms" 
 */
static int
method_kittler ( const Image *img )
{
 SET_FUNC_NAME ( "method_kittler" );
 int threshold;
 int init_threshold;
 int t_prev = -1;
 int ih;
 int iter = 0;
 double p, q;
 double mu, nu; 
 double sigma2, tau2;
 double log_term;
 double w0, w1, w2;
 double discriminant;
 double temp;
 double A_max, B_max, C_max;
 double *data;
 double A[NUM_GRAY];
 double B[NUM_GRAY];
 double C[NUM_GRAY];
 Histo *histo;
 Histo *norm_histo;
  
 if ( !is_gray_img ( img ) )
  {
   ERROR_RET ( "Not a grayscale image !", INT_MIN );
  }

 /* Calculate the histogram */
 histo = create_histo ( img );

 /* Calculate the normalized histogram */
 norm_histo = normalize_histo ( histo );

 data = get_histo_data ( norm_histo );

 A[0] = data[0];
 B[0] = 0.0;
 C[0] = 0.0;
 for ( ih = 0 + 1; ih < NUM_GRAY; ih++ )
  {
   A[ih] = A[ih - 1] + data[ih];
   B[ih] = B[ih - 1] + ih * data[ih];
   C[ih] = C[ih - 1] + ih * ih * data[ih];
  }

 A_max = A[MAX_GRAY];
 B_max = B[MAX_GRAY];
 C_max = C[MAX_GRAY];
 
 init_threshold = threshold_iter ( img );
 threshold = init_threshold;

 while ( threshold != t_prev )
  {
   /* Probability of background */
   p = A[threshold] / A_max;
   /* Probability of object */
   q = 1.0 - p;

   /* Mean of background */
   mu = B[threshold] / A[threshold];
   /* Mean of object */
   nu = ( B_max - B[threshold] ) / ( A_max - A[threshold] );

   /* Variance of background */
   sigma2 = C[threshold] / A[threshold] - mu * mu;
   /* Variance of object */
   tau2 = ( C_max - C[threshold] ) / ( A_max - A[threshold] ) - nu * nu;

   /* The terms of the quadratic equation to be solved */
   w0 = 1.0 / sigma2 - 1.0 / tau2;
   w1 = mu / sigma2 - nu / tau2;
   log_term = ( sigma2 * q * q ) / ( tau2 * p * p );

   if ( IS_NEG ( log_term ) || IS_ZERO ( log_term ) ) 
    {
     fprintf ( stderr, "Algorithm not converging [log_term = %f]\n", log_term );
     /* Pathological case: return the Ridler & Calvard threshold */
     return init_threshold;
    }

   w2 = ( mu * mu ) / sigma2 - ( nu * nu ) / tau2 + log ( log_term );

   discriminant = w1 * w1 - w0 * w2;
   if ( IS_NEG ( discriminant ) )
    {
     fprintf ( stderr, "Algorithm not converging [discriminant = %f]\n", discriminant );
     /* Pathological case: return the Ridler & Calvard threshold */
     return init_threshold;
    }

   /* The updated threshold is the integer part of the solution of the quadratic equation. */
   t_prev = threshold;

   temp = ( w1 + sqrt ( discriminant ) ) / w0;
   if ( isnan ( temp ) )
    {
     fprintf ( stderr, "Algorithm not converging [temp = %f]\n", temp );
     /* Pathological case: return the Ridler & Calvard threshold */
     return init_threshold;
    }

   threshold = temp + 0.5; /* round */

   /* printf ( "Iter #%d: temp = %f ; Threshold = %d\n", iter, temp, threshold ); */
   iter++;
  }

 free_histo ( histo );
 free_histo ( norm_histo );

 return threshold;
}

static int
method_otsu ( const Image * img )
{
 SET_FUNC_NAME ( "method_otsu" );
 int ih;
 int threshold;
 int first_bin;			/* see below */
 int last_bin;			/* see below */
 double total_mean;		/* mean gray-level for the whole image */
 double bcv;			/* between-class variance */
 double max_bcv;		/* max BCV */
 double *cnh;			/* cumulative normalized histogram */
 double *mean;			/* mean gray-level */
 double *data;			/* normalized histogram data */
 Histo *histo;		        /* histogram */
 Histo *norm_histo;		/* normalized histogram */

 if ( !is_gray_img ( img ) )
  {
   ERROR_RET ( "Not a grayscale image !", INT_MIN );
  }

 /* Calculate the histogram */
 histo = create_histo ( img );

 /* Calculate the normalized histogram */
 norm_histo = normalize_histo ( histo );

 data = get_histo_data ( norm_histo );

 /* Calculate the cumulative normalized histogram */
 cnh = accumulate_histo ( norm_histo );

 mean = ( double * ) malloc ( NUM_GRAY * sizeof ( double ) );

 mean[0] = 0.0;
 for ( ih = 0 + 1; ih < NUM_GRAY; ih++ )
  {
   mean[ih] = mean[ih - 1] + ih * data[ih];
  }

 total_mean = mean[MAX_GRAY];

 /* 
    Determine the first non-zero 
    bin starting from the first bin 
  */
 first_bin = 0;
 for ( ih = 0; ih < NUM_GRAY; ih++ )
  {
   if ( !IS_ZERO ( cnh[ih] ) )
    {
     first_bin = ih;
     break;
    }
  }

 /* 
    Determine the first non-one bin 
    starting from the last bin
  */
 last_bin = MAX_GRAY;
 for ( ih = MAX_GRAY; ih >= first_bin; ih-- )
  {
   if ( !IS_ZERO ( 1.0 - cnh[ih] ) )
    {
     last_bin = ih;
     break;
    }
  }

 /* 
    Calculate the BCV at each gray-level and
    find the threshold that maximizes it 
  */
 threshold = INT_MIN;
 max_bcv = 0.0;
 for ( ih = first_bin; ih <= last_bin; ih++ )
  {
   bcv = total_mean * cnh[ih] - mean[ih];
   bcv *= bcv / ( cnh[ih] * ( 1.0 - cnh[ih] ) );

   if ( max_bcv < bcv )
    {
     max_bcv = bcv;
     threshold = ih;
    }

   /* fprintf ( stderr, "T = %d ; BCV = %f\n", ih, bcv ); */
  }

 free_histo ( histo );
 free_histo ( norm_histo );
 free ( cnh );
 free ( mean );

 return threshold;
}

static Image *
threshold_fusion ( const Image *in_img, const int win_size, const double gamma )
{
 SET_FUNC_NAME ( "threshold_fusion" );
 byte **in_data;
 byte **out_data;
 int num_rows, num_cols;
 int half_win;
 int win_count;			/* number of pixels in the filtering window */
 int ik;
 int it;
 int ir, ic;
 int iwr, iwc;
 int r_begin, r_end;		/* vertical limits of the filtering operation */
 int c_begin, c_end;		/* horizontal limits of the filtering operation */
 int wr_begin, wr_end;		/* vertical limits of the filtering window */
 int wc_begin, wc_end;		/* horizontal limits of the filtering window */
 int index;
 int threshold;
 int count_obj;
 int thresh[NUM_METHODS];
 int *win_data;
 double thresh_mean;
 double U_II_obj;
 double U_II_back;
 double alpha_sum;
 double *alpha_lut;
 double *beta;
 int ( *func[NUM_METHODS] ) ( const Image * ) = { &method_huang, &method_kapur, 
	                                          &method_kittler, &method_otsu };
 Image *out_img;

 if ( !is_gray_img ( in_img ) )
  {
   ERROR_RET ( "Not a grayscale image !", NULL );
  }

 if ( !IS_POS_ODD ( win_size ) )
  {
   ERROR ( "Window size ( %d ) must be positive and odd !", win_size );
   return NULL;
  }

 half_win = win_size / 2;
 win_count = win_size * win_size;

 beta = ( double * ) calloc ( NUM_METHODS, sizeof ( double ) );
 alpha_lut = ( double * ) calloc ( NUM_GRAY, sizeof ( double ) );
 win_data = ( int * ) calloc ( win_count, sizeof ( int ) );

 num_rows = get_num_rows ( in_img );
 num_cols = get_num_cols ( in_img );
 in_data = get_img_data_nd ( in_img );

 out_img = alloc_img ( PIX_BIN, num_rows, num_cols );
 if ( IS_NULL ( out_img ) )
  {
   ERROR_RET ( "Insufficient memory !", NULL );
  }

 out_data = get_img_data_nd ( out_img );

 thresh_mean = 0.0;
 for ( it = 0; it < NUM_METHODS; it++ )
  {
   thresh[it] = CLAMP_BYTE ( func[it] ( in_img ) );
   thresh_mean += thresh[it];
  }

 thresh_mean /= NUM_METHODS;
 
 if ( NUM_METHODS == 1 )
  {
   return negate_img ( threshold_img ( in_img, thresh[0] ) );
  }
 
 for ( it = 0; it < NUM_METHODS; it++ )
  {
   beta[it] = exp ( -gamma * fabs ( thresh_mean - thresh[it] ) );
  }

 for ( ik = 0; ik < NUM_GRAY; ik++ )
  {
   alpha_lut[ik] = 1.0 - exp ( -gamma * ik );
  }

 /* 
    Determine the limits of the filtering operation. Pixels
    in the output image outside these limits are set to 0.
  */
 r_begin = half_win;
 r_end = num_rows - half_win;
 c_begin = half_win;
 c_end = num_cols - half_win;

 /* Initialize the vertical limits of the filtering window */
 wr_begin = 0;
 wr_end = win_size;

 /* For each image row */
 for ( ir = r_begin; ir < r_end; ir++ )
  {
   /* Initialize the horizontal limits of the filtering window */
   wc_begin = 0;
   wc_end = win_size;

   /* For each image column */
   for ( ic = c_begin; ic < c_end; ic++ )
    {
     index = 0;
     for ( iwr = wr_begin; iwr < wr_end; iwr++ )
      {
       for ( iwc = wc_begin; iwc < wc_end; iwc++ )
        {
         win_data[index++] = in_data[iwr][iwc];
	}
      }

     U_II_obj = U_II_back = 0.0;
     for ( it = 0; it < NUM_METHODS; it++ )
      {
       count_obj = 0;
       threshold = thresh[it];
       alpha_sum = 0.0;

       for ( ik = 0; ik < win_count; ik++ )
        {
	 count_obj += win_data[ik] > threshold;
	 alpha_sum += alpha_lut[abs ( win_data[ik] - threshold )];
	}

       U_II_obj += beta[it] * alpha_sum * count_obj;
       U_II_back += beta[it] * alpha_sum * ( win_count - count_obj );
      }

     /* Object is darker than the background */
     out_data[ir][ic] = U_II_obj <= U_II_back;

     /* Update the horizontal limits of the filtering window */
     wc_begin++;
     wc_end++;
    }

   /* Update the vertical limits of the filtering window */
   wr_begin++;
   wr_end++;
  }

 free ( beta );
 free ( alpha_lut );
 free ( win_data );

 return out_img;
}

static void
calc_border_error ( const Image *auto_border_img, const Image *man_border_img )
{
 SET_FUNC_NAME ( "calc_border_error" );
 byte *auto_data;
 byte *man_data;
 int ik;
 int num_pixels;
 int man_border_area, auto_border_area;
 double true_pos, false_neg, false_pos, true_neg;
 double xor_error, precision, recall, specificity;

 if ( !is_bin_img ( auto_border_img ) || !is_bin_img ( man_border_img ) )
  {
   ERROR ( "Input images must be binary !" );
  }

 if ( !img_dims_agree ( auto_border_img, man_border_img ) )
  {
   ERROR ( "Input images must have the same dimensions !" );
  }

 num_pixels = get_num_rows ( auto_border_img ) * get_num_cols ( auto_border_img );

 auto_data = get_img_data_1d ( auto_border_img );
 man_data = get_img_data_1d ( man_border_img );

 true_pos = false_neg = false_pos = true_neg = 0.0;

 for ( ik = 0; ik < num_pixels; ik++ )
  {
   if ( man_data[ik] == OBJECT )
    {
     if ( auto_data[ik] == OBJECT )
      {
       true_pos += 1.0;
      }
     else
      {
       false_neg += 1.0;
      }
    }
   else
    {
     if ( auto_data[ik] == BACKGROUND )
      {
       true_neg += 1.0;
      }
     else
      {
       false_pos += 1.0;   
      }
    }
  }

 man_border_area = true_pos + false_neg;
 auto_border_area = true_pos + false_pos;

 xor_error = 100.0 * ( ( false_pos + false_neg ) / man_border_area );
 precision = 100.0 * ( true_pos / auto_border_area );
 recall = 100.0 * ( true_pos / man_border_area ); /* sensitivity */
 specificity = 100.0 * ( true_neg / ( false_pos + true_neg ) );

 if ( VERBOSE )
  {
   printf ( "xor = %f\nprecision = %f\nrecall (sensitivity) = %f\nspecificity = %f\n", 
            xor_error, precision, recall, specificity );
  }
}

static Image *
fill_holes_fast ( const Image *in_img )
{
 SET_FUNC_NAME ( "fill_holes_fast" );
 int num_rows, num_cols;
 int num_pixels;
 int ik;
 int num_cc;
 int last_row_idx, last_col_idx;
 int **lab_data_2d;
 int *lab_data_1d;
 byte *out_data;
 Bool *is_hole;
 Image *neg_img;
 Image *lab_img;
 Image *out_img;

 if ( !is_bin_img ( in_img ) )
  {
   ERROR_RET ( "Not a binary image !", NULL );
  }

 num_rows = get_num_rows ( in_img );
 num_cols = get_num_cols ( in_img );
 num_pixels = num_rows * num_cols;

 /* Clone the input image whose holes will be filled later */
 out_img = clone_img ( in_img );
 out_data = get_img_data_1d ( out_img );

 /* Negate the input image and label its connected components */
 neg_img = negate_img ( in_img );
 lab_img = label_cc ( neg_img, 4 );

 lab_data_1d = get_img_data_1d ( lab_img );
 lab_data_2d = get_img_data_nd ( lab_img );

 num_cc = get_num_cc ( lab_img );

 is_hole = ( Bool * ) malloc ( ( num_cc + 1 ) * sizeof ( Bool ) );

 /* Initially every component is assumed to be a hole */
 for ( ik = 0; ik <= num_cc; ik++ ) 
  {
   is_hole[ik] = true;
  }

 /* Components that touch the top and bottom image walls are not holes */
 last_row_idx = num_rows - 1;
 for ( ik = 0; ik < num_cols; ik++ )
  {
   /* First row */
   is_hole[lab_data_2d[0][ik]] = false;

   /* Last row */
   is_hole[lab_data_2d[last_row_idx][ik]] = false;
  }

 /* Components that touch the left and right image walls are not holes */
 last_col_idx = num_cols - 1;
 for ( ik = 0; ik < num_rows; ik++ )
  {
   /* First column */
   is_hole[lab_data_2d[ik][0]] = false;

   /* Last column */
   is_hole[lab_data_2d[ik][last_col_idx]] = false;
  }

 /* Background component cannot be a hole */
 is_hole[0] = false;

 for ( ik = 0; ik < num_pixels; ik++ )
  {
   /* Change the hole pixels to OBJECT */
   if ( is_hole[lab_data_1d[ik]] )
    {
     out_data[ik] = OBJECT;
    }
  }

 free_img ( neg_img );
 free_img ( lab_img );
 free ( is_hole );

 return out_img;
}

static PointList *
overlay_contour ( const Image *in_img, const Image *bin_img, const Color color )
{
 Chain *chain;
 PointList *contour;
 Image *label_img;

 /* Determine the contour of BIN_IMG */
 label_img = label_cc ( bin_img, 4 );
 chain = trace_contour ( label_img, OBJECT, 65536 );
 contour = chain_to_point_list ( chain );

 if ( is_rgb_img ( in_img ) )
  {
   byte *in_data;
   byte *over_data;
   int num_rows, num_cols;
   int num_pixels_t3;
   int i, j;
   Image *over_img;

   num_rows = get_num_rows ( in_img );
   num_cols = get_num_cols ( in_img );
   num_pixels_t3 = 3 * num_rows * num_cols;
   in_data = get_img_data_1d ( in_img );

   /* Overlay the contour on IN_IMG */
   over_img = point_list_to_img ( contour, num_rows, num_cols );
   over_data = get_img_data_1d ( over_img );

   for ( i = j = 0; i < num_pixels_t3; j++, i += 3 )
    {
     if ( over_data[j] == OBJECT )
      {
       in_data[i] = color.red;
       in_data[i + 1] = color.green;
       in_data[i + 2] = color.blue;
      }
    }
 
   free_img ( over_img );
  }

 free_img ( label_img );
 free_chain ( chain );

 return contour;
}

int
main ( int argc, char ** argv )
{
 char in_file_name[256];
 char border_file_name[256];
 char out_file_name[256];
 int radius;
 int dilation_factor = 0;
 int median_win_size = 0;
 int fusion_win_size = 3;
 Color red = { 255, 0, 0 };
 Color green = { 0, 255, 0 };
 Color blue = { 0, 0, 255 };
 clock_t start_time;
 double gamma = 0.1;
 double elapsed_time;
 double diameter;
 PointList *contour;
 PointList *convex_hull;
 Strel *se;
 Image *in_img;
 Image *label_img;
 Image *mono_img;
 Image *tmp_label_img;
 Image *tmp_bin_img;
 Image *border_img;
 Image *out_img;
 
 if ( argc < 2 )
  {
   fprintf ( stderr, "Usage: %s [-dilate <value>] [-median <value>] [-fusion <value>] [-gamma <value>] <RGB or grayscale image>\n", argv[0] );
   fprintf ( stderr, "\t-dilate <value>: Dilation factor [default = 0]\n" );
   fprintf ( stderr, "\t-median <value>: Median filter size [default = 0]\n" );
   fprintf ( stderr, "\t-fusion <value>: Fusion window size [default = 3]\n" );
   fprintf ( stderr, "\t-gamma <value>: Fusion gamma value [default = 0.1]\n" );
   exit ( EXIT_FAILURE );
  }

 /* Parse the arguments */
 for ( argv++, argc--; argc >= 2; argv += 2, argc -= 2 )
  {
   if ( !strcmp ( argv[0], "-dilate" ) )
    {
     dilation_factor = atoi ( argv[1] );
    }
   else if ( !strcmp ( argv[0], "-median" ) )
    {
     median_win_size = atoi ( argv[1] );
    }
   else if ( !strcmp ( argv[0], "-fusion" ) )
    {
     fusion_win_size = atoi ( argv[1] );
    }
   else if ( !strcmp ( argv[0], "-gamma" ) )
    {
     gamma = atof ( argv[1] );
    }
  }

 /* Read the input image */
 strcpy ( in_file_name, argv[0] );
 in_img = read_img ( in_file_name );

 printf ( "File: %s\n", in_file_name );

 if ( is_rgb_img ( in_img ) )
  {
   Image *red_img, *green_img, *blue_img;

   get_rgb_bands ( in_img, &red_img, &green_img, &blue_img );
   mono_img = blue_img;

   free_img ( red_img );
   free_img ( green_img );
  } 
 else if ( is_gray_img ( in_img ) )
  {
   mono_img = clone_img ( in_img );
  }
 else
  {
   fprintf ( stderr, "Input image (%s) must be RGB or grayscale !\n", in_file_name );
   exit ( EXIT_FAILURE );
  }

 start_time = start_timer ( );

 /* Perform median filtering */
 if ( median_win_size > 1 )
  {
   Image *tmp_gray_img;
   
   tmp_gray_img = filter_running_median ( mono_img, median_win_size );
   free_img ( mono_img );
   mono_img = clone_img ( tmp_gray_img );
  }

 /* Perform threshold fusion */
 out_img = threshold_fusion ( mono_img, fusion_win_size, gamma );
 free_img ( mono_img );

 /* Eliminate all but the largest connected component */
 tmp_label_img = label_cc ( out_img, 4 );
 label_img = retain_largest_cc ( tmp_label_img );

 free_img ( out_img );
 free_img ( tmp_label_img );

 /* Fill the holes */
 tmp_bin_img = label_to_bin ( label_img  );
 out_img = fill_holes_fast ( tmp_bin_img );

 free_img ( label_img );
 free_img ( tmp_bin_img );

 /* Overlay the initial contour on the original image (red) */
 contour = overlay_contour ( in_img, out_img, red );
 
 /* Determine the lesion diameter */
 convex_hull = calc_2d_convex_hull ( contour );
 diameter = calc_max_diameter ( convex_hull );

 free_point_list ( contour );
 free_point_list ( convex_hull );

 if ( dilation_factor > 0 )
  {
   /* Determine the structuring element radius */
   radius = dilation_factor * 0.5 * diameter / 256.0 + 0.5; /* round */
   se = make_disk_strel ( radius );
 
   /* Perform morphological dilation */
   tmp_bin_img = dilate_img ( out_img, se );
   free_nd ( se->data_2d, 2 );
   free_img ( out_img );
   out_img = clone_img ( tmp_bin_img );
   free_img ( tmp_bin_img );
  }

 elapsed_time = stop_timer ( start_time );
 
 in_file_name[strlen ( in_file_name ) - 4] = '\0';

 /* Write the binary output file */
 sprintf ( out_file_name, "out_%s_bin.bmp", in_file_name );
 write_img ( out_img, out_file_name, FMT_BMP );

 /* Obtain the final contour */
 if ( dilation_factor > 0 )
  {
   contour = overlay_contour ( in_img, out_img, green );
   free_point_list ( contour );
  }
   
 /* Read the manual border image */
 sprintf ( border_file_name, "%sborder_%s.pbm", BORDER_PATH, in_file_name );
 set_err_mode ( false );
 border_img = read_img ( border_file_name );
 set_err_mode ( true );

 if ( border_img != NULL )
  {
   /* Calculate the XOR measure */
   calc_border_error ( out_img, border_img );
 
   /* Overlay the manual contour on the original image (blue) */
   contour = overlay_contour ( in_img, border_img, blue );
   free_point_list ( contour );
  }

 /* Write the RGB output file */
 if ( is_rgb_img ( in_img ) )
  {
   sprintf ( out_file_name, "out_%s.bmp", in_file_name );
   write_img ( in_img, out_file_name, FMT_BMP );
  }

 free_img ( out_img );
 
 printf ( "CPU time = %.0f msecs.\n", 1000 * elapsed_time );

 return 0;
}

/* CHECK FOR MEMORY LEAKS WITH VALGRIND */
/* TODO: EXPLAIN THE PARAMETERS IN MORE DETAIL */
/* return negate_img ( threshold_img ( in_img, thresh[0] ) ); */


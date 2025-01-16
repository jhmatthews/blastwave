#define  PHYSICS                        RHD
#define  DIMENSIONS                     2
#define  GEOMETRY                       SPHERICAL
#define  BODY_FORCE                     NO
#define  COOLING                        NO
#define  RECONSTRUCTION                 LINEAR
#define  TIME_STEPPING                  RK2
#define  NTRACER                        2
#define  PARTICLES                      NO
#define  USER_DEF_PARAMETERS            9

/* -- physics dependent declarations -- */

#define  EOS                            TAUB
#define  ENTROPY_SWITCH                 NO
#define  RADIATION                      NO

/* -- user-defined parameters (labels) -- */

#define  RHO_BLOB                       0
#define  RHO_ISM                        1
#define  GAMMA                          2
#define  BLOB_RAD                       3
#define  CS_ISM                         4
#define  LZ_FACT                        5
#define  PRS_BLOB                       6
#define  KAPPA                          7
#define  PHI_DEGREES                    8

/* [Beg] user-defined constants (do not change this line) */

#define  UNIT_DENSITY                   1.672661e-24
#define  LIMITER                        MC_LIM
#define  UNIT_VELOCITY                  CONST_c
#define  UNIT_LENGTH                    1e15
#define  SHOCK_FLATTENING               MULTID

/* [End] user-defined constants (do not change this line) */

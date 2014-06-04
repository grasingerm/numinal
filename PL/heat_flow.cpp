#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
    const double L = 1.0;
    const double hp_to_kA = 25.0;
    
    const double delta_x = 0.2;
    
    const double T_B = 100.0;
    const double T_inf = 20.0;
    
    //TODO: write this for an arbitrary sized mesh
    
    /* nodes 2 thru 4 */
    double a_w = 1/delta_x;
    double a_e = 1/delta_x;
    double S_p = -hp_to_kA * delta_x;
    double S_u = hp_to_kA * delta_x * T_inf;
    double a_p = a_w + a_e - S_p;
    
    mat::fixed<5,5> A;
    A << a_e + hp_to_kA*delta_x + 2/delta_x << -a_e << 0 << 0 << 0 << endr
      << -a_w << a_w + a_e + hp_to_kA*delta_x << -a_e << 0 << 0 << endr
      << 0 << -a_w << a_w + a_e + hp_to_kA*delta_x << -a_e << 0 << endr
      << 0 << 0 << -a_w << a_w + a_e + hp_to_kA*delta_x << -a_e << endr
      << 0 << 0 << 0 << -a_w << a_w + hp_to_kA*delta_x << endr;
      
    vec::fixed<5> S;
    S << hp_to_kA * delta_x * T_inf + 2/delta_x * T_B << endr
      << hp_to_kA * delta_x * T_inf << endr
      << hp_to_kA * delta_x * T_inf << endr
      << hp_to_kA * delta_x * T_inf << endr
      << hp_to_kA * delta_x * T_inf << endr;
      
    vec::fixed<5> T;
    
    solve(T, A, S);
    
    cout << "A = " << endl << A << endl;
    cout << "S = " << endl << S << endl;
    cout << "T = " << endl << T << endl;

    return 0;
}

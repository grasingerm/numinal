#include <iostream>
#include <list>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;

class Polynomial
{
private:
    list<double> coeffs;
public:
    Polynomial() {};
    void add_coeff(const double coeff) { coeffs.push_front(coeff); }
    int order() { return coeffs.size()-1; }
    double evaluate(const double x)
    {
        double u = 0;
        for (auto a_n : coeffs)
            u = u*x + a_n;
        return u;
    }
    list<double> get_coeffs() { return coeffs; }
};

ostream& operator<< (ostream& os, Polynomial& poly)
{
    int o = poly.order();
    auto coeff = poly.get_coeffs().begin();
    if (o > 0)
    {
        os << *coeff << "x^" << o;
        o--;
        coeff++;
    }
    for (auto end = poly.get_coeffs().end(); coeff != end; coeff++)
    {
        if (o > 0)
            os << " + " << *coeff << "x^" << o--;
        else
            os << " + " << *coeff;
    }
    return os;
}

int main(int argc, char* argv[])
{
    Polynomial poly;
    double coeff, x;
    string do_continue = "y";
    
    while (do_continue.front() == 'y')
    {
        while (1)
        {
            cout << "Enter a coefficient (ascending order, -999 to quit) ";
            cin >> coeff;
            
            if (coeff == -999) break;
            poly.add_coeff(coeff);
        }
        
        //cout << poly << endl;
        
        while (1)
        {
            cout << "Enter an x value (-999 to quit) ";
            cin >> x;
            
            if (x == -999) break;
            cout << "f(" << x << ") = " << poly.evaluate(x) << endl;
        }
        
        cout << "exited loops" << endl;
        cin >> do_continue;
        cout << do_continue << endl;
        boost::algorithm::to_lower(do_continue);
        cout << do_continue << endl;
    }

    return 0;
}

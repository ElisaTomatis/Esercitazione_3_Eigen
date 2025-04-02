#include <iostream>
#include <Eigen/Dense>

#include "LinearSolver.hpp"

using namespace std;
using namespace Eigen;

// Risolve Ax = b usando la decomposizione PALU
VectorXd solvePALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

// Risolve Ax = b usando la decomposizione QR
VectorXd solveQR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// Calcola l'errore relativo tra la soluzione trovata e quella attesa
double relativeError(const VectorXd& x, const VectorXd& x_exact) {
    return (x - x_exact).norm() / x_exact.norm();
}


double conditionNumber(const MatrixXd& A) {
    JacobiSVD<MatrixXd> svd(A);
    return svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
}

int main() {
    vector<MatrixXd> A_matrices = {
        (MatrixXd(2,2) << 0.5547001962252291, -0.03770900990025203, 0.8320502943378437, -0.9992887623566787).finished(),
        (MatrixXd(2,2) << 0.5547001962252291, -0.5540607316466765, 0.320502943378437, -0.83247624929913135).finished(),
        (MatrixXd(2,2) << 0.5547001962252291, -0.5547001955851905, 0.8320502943378437, -0.832050294764536).finished()
    };
    
    vector<VectorXd> b_vectors = {
        (VectorXd(2) << -0.5169911863249772, 0.1672384680188350).finished(),
        (VectorXd(2) << -0.0006394645785530173, 0.0004259549612877223).finished(),
        (VectorXd(2) << -6.400391328043042e-10, 4.266924591433963e-10).finished()
    };
    
    VectorXd x_exact(2);
    x_exact << -1.0, -1.0;
    
    for (size_t i = 0; i < A_matrices.size(); ++i) {
        cout << "Solving system " << i+1 << " using PALU and QR decomposition" << endl;
        
        double condA = conditionNumber(A_matrices[i]);
        
        if (condA > 1e6) {
			cout << "Condition number: " << condA << endl;
            cout << "Warning: Matrix is ill-conditioned, results may be inaccurate." << endl;
			return 1;
		}
		
		else {
			VectorXd x_palu = solvePALU(A_matrices[i], b_vectors[i]);
			VectorXd x_qr = solveQR(A_matrices[i], b_vectors[i]);
			
			cout << "PALU Solution: \n" << x_palu.transpose() << endl;
			cout << "QR Solution: \n" << x_qr.transpose() << endl;
			
			cout << "PALU Relative Error: " << relativeError(x_palu, x_exact) << endl;
			cout << "QR Relative Error: " << relativeError(x_qr, x_exact) << endl;
     
        }
		cout << "\n" << endl;
    }
    
    return 0;
}

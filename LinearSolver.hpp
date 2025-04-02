#pragma once

#include <Eigen/Dense>

using namespace Eigen;

// Risolve Ax = b usando la decomposizione PALU
VectorXd solvePALU(const MatrixXd& A, const VectorXd& b);

// Risolve Ax = b usando la decomposizione QR
VectorXd solveQR(const MatrixXd& A, const VectorXd& b);

// Calcola l'errore relativo tra la soluzione trovata e quella attesa
double relativeError(const VectorXd& x, const VectorXd& x_exact);
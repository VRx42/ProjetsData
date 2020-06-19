function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_x = X * theta; %(12,1)
J = (1 /(2*m)) * sum((h_x - y).^2); %12,1

reg_term  = (lambda / ( 2 * m)) * sum(theta(2:end).^2);
J = J + reg_term;

grad(1) = (1/m) * X(:,1)' * (h_x - y); %X(:,1)on prend toutes les lignes de la premiere colonne et on la transpose, ça nous donne tous les xi
grad(2:end) = (1/m) * X(:,2:end)' * (h_x - y) + (lambda/m)*theta(2:end);%pour la colonne deux du grad on a que la colonne X(:2) qui se somme, parce que ce sont les features associées a 2e theta
% =========================================================================

grad = grad(:);

end

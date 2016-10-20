require 'daru'
require 'nmatrix'
require 'nyaplot'
require 'nyaplot3d'
require 'pp'
require 'gsl'

include GSL::MultiMin
include GSL::Stats

def sigmoid(elmt)
  return 1 / (1 + Math::exp(-elmt))
end

def sigmoid_matrix(m)
  g = GSL::Matrix.zeros(m.shape[0], 1)

  i = 0
  m.each_row do |row|
    j = 0
    row.each do |elmt|
      g[i, j] = sigmoid(elmt)
      j = j + 1
    end
    i = i + 1
  end

  return g
end

def log_matrix(m)
  return m.collect { |item| item = Math.log(item) }
end

def cost_function(theta, x, y)
  m = y.shape[0].to_f
  h = sigmoid_matrix(x * theta)
  j = (1.0/m) * ((-y.transpose * log_matrix(h)) - ((1-y).transpose * log_matrix(1-h)))

  return j
end

def gradient_descent(theta, x, y)
  m = y.shape[0].to_f
  h = sigmoid_matrix(x * theta)
  grad = (1.0/m) * (x.transpose * (h-y))

  return grad
end

def mean_matrix(x)
  m = x.shape[0]
  return GSL::Matrix[[GSL::Stats.mean(x[0..(m-1), 0]), GSL::Stats.mean(x[0..(m-1), 1])]]
end

def sd_matrix(x)
  m = x.shape[0]
  return GSL::Matrix[[x[0..(m-1), 0].sd, x[0..(m-1), 1].sd]]
end

def divide_matrix(x, sigma)
  m = x.shape[0]
  return GSL::Matrix[x[0..(m-1), 0]/sigma[0], x[0..(m-1), 1]/sigma[1]]
end

def feature_normalize(x)
  m = x.shape[0]
  mu = mean_matrix(x)
  sigma = sd_matrix(x)
  x_min_mu = GSL::Matrix[x[0..(m-1), 0]-mu[0], x[0..(m-1), 1]-mu[1]]
  x_normalized = divide_matrix(x_min_mu, sigma)
  return x_normalized, mu, sigma
end

def gdbt(theta, x, y, alpha, beta, iter, tol)
  # gdbt: Gradient Descent BackTracking
  # Gradient descent minimization based on backtracking line search
  # Output: Minimizer within precision or maximum iteration number
  # Input:
      # theta: Initial value
      # X: Training data (input)
      # y: Training data (output)
      # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
      # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
      # iter: Maximum number of iterations
      # tol: The procedure will break if the square of the 2-norm of the gradient is less than the threshold tol
  for i in 0..iter do
    grad = gradient_descent(theta, x, y)
    delta = -grad
    if (grad.transpose * grad)[0] < tol
      puts "Terminated due to stopping condition with iteration number" + i.to_s
      return theta
    end

    j = cost_function(theta, x, y)
    alpha_grad_delta = alpha * grad.transpose * delta

    # begin line search
    t = 1
    while cost_function(theta+t * delta, x, y)[0] > (j + t * alpha_grad_delta)[0] do
      t = beta * t
    end
    # end line search

    # update
    theta = theta + t * delta
  end

  return theta
end

def predict(theta, x)
  i = 0
  prediction = GSL::Matrix.alloc(100, 1)
  x.each_row do |row|
    if sigmoid((row.matrix_view(1, 3) * theta)[0]) >= 0.5
      prediction[i] = 1
    else
      prediction[i] = 0
    end
    i += 1
  end

  return prediction
end

def plot_data(plt, x, y)
  pos_x = []
  pos_y = []
  neg_x = []
  neg_y = []

  i = 0
  y.each_row do |row|
    if row[0] == 1.0
      pos_x << x[i,0]
      pos_y << x[i,1]
    elsif row[0] == 0.0
      neg_x << x[i, 0]
      neg_y << x[i, 1]
    end
    i += 1
  end

  plt.x_label('Exam 1')
  plt.y_label('Exam 2')

  pos = plt.add(:scatter, pos_x, pos_y)
  pos.title("Admitted")
  pos.color("blue")

  neg = plt.add(:scatter, neg_x, neg_y)
  neg.title("Not Admitted")
  neg.color("yellow")

  plt.legend(true)
  plt.show
end

data = []
File.open('/opt/ruby-projects/machine-learning/ex2data1.txt').readlines.each do |line|
  row_data = line.chomp.split(',')
  data[0].nil? ? data[0] = [row_data[0].to_f] : data[0] << row_data[0].to_f
  data[1].nil? ? data[1] = [row_data[1].to_f] : data[1] << row_data[1].to_f
  data[2].nil? ? data[2] = [row_data[2].to_f] : data[2] << row_data[2].to_f
end
x = GSL::Matrix[data[0], data[1]].transpose
y = GSL::Matrix[data[2]].transpose

plt = Nyaplot::Plot.new
plot_data(plt, x, y)

m = x.shape[0]
n = x.shape[1]

initial_theta = GSL::Matrix.zeros(3,1)
x_matrix = GSL::Matrix.ones(m, 1).horzcat(x)

cost_zero = cost_function(initial_theta, x_matrix, y)
gradient_zero = gradient_descent(initial_theta, x_matrix, y)

puts "Cost at initial theta (zeros): " + cost_zero.to_s
puts "Gradient at initial theta (zeros): "
pp gradient_zero

normalized = feature_normalize(x)
mu = normalized[1]
sigma = normalized[2]
x_normalized = normalized[0]
x_normalized = GSL::Matrix.ones(m, 1).horzcat(x_normalized)

alpha = 0.01
beta = 0.8
theta_unnorm = gdbt(initial_theta, x_normalized, y, alpha, beta, 1000, 1e-8)
theta = theta_unnorm
theta[0] = theta[0] - ((GSL::Matrix[[theta_unnorm[1]], [theta_unnorm[2]]].mul_elements(mu.transpose))/sigma.transpose).to_vview.sum
for i in 1..(theta.size[0]-1) do
  theta[i] = theta_unnorm[i]/sigma[0,i-1]
end

prediction = predict(theta, x_matrix)
puts "Cost at theta found by gdbt: "
puts cost_function(theta, x_matrix, y)
puts "theta: "
pp theta


score1 = GSL::Vector.linspace(30, 100, 200).to_a.chunk{|n| n}.map(&:first)
score2 = ((-theta[0] - (GSL::Matrix[score1].transpose * theta[1]))/theta[2]).to_a.flatten.chunk{|n| n}.map(&:first)

puts "For a student with scores 45 and 85, we predict an admission " + sigmoid_matrix(GSL::Matrix[[1,45,85]] * theta)[0].to_s
puts "Train accuracy: " + (y.size[0]-((y - prediction).abs).to_vview.sum).to_s

plt.add(:line, score1, score2)
plt.show
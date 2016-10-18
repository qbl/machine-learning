require 'daru'
require 'nmatrix'
require 'nyaplot'
require 'nyaplot3d'
require 'pp'
require 'gsl'

# compute_cost should return a single value, ex: j=0
def compute_cost(x, y, theta)
  m = y.size
  hyphothesis = x.dot theta
  sigma = (hyphothesis-y) ** 2
  j = 1/(2*m).to_f * (sigma.sum)[0]

  return j
end

def gradient_descent(x, y, theta, alpha, iterations)
  j_history = []
  for i in 0..iterations do
    m = y.size
    hyphothesis = x.dot theta
    x_features = x[0..m-1, 1]

    theta_temp0 = theta[0] - alpha*((hyphothesis-y).sum)[0]/m
    theta_temp1 = theta[1] - alpha*(((hyphothesis-y).transpose).dot x_features)[0]/m
    theta = N[[theta_temp0], theta_temp1]
    j_history[i] = compute_cost(x, y, theta)
  end

  return theta, j_history
end

# Part 2: Plotting
x = []
y = []
File.open('/opt/ruby-projects/machine-learning/ex1data1.txt').readlines.each do |line|
  data = line.chomp.split(',')
  x << data[0].to_f
  y << data[1].to_f
end

plt = Nyaplot::Plot.new
plt.x_label('x')
plt.y_label('y')
plt.add(:scatter, x, y)
plt.show

# Part 3: Gradient descent
alpha = 0.01;
iterations = 1500
theta = N[[0],[0]]

m = y.size
ones_matrix = NMatrix.new([m, 1], 1.0)
x_nmatrix = ones_matrix.concat(NMatrix.new([m, 1], x))
y_nmatrix = NMatrix.new([m, 1], y)

puts 'Test compute_cost function: cost = ' + compute_cost(x_nmatrix, y_nmatrix, theta).to_s

a = gradient_descent(x_nmatrix, y_nmatrix, theta, alpha, iterations)

theta = a[0]
j_history = a[1]
puts 'Theta found by gradient descent:'
puts theta

plt = Nyaplot::Plot.new
plt.add(:line, (1..j_history.size).to_a, j_history)
plt.y_label('Cost over iteration')
plt.x_label('Number of iteration')
plt.show

x_by_theta = x_nmatrix.dot theta

plt = Nyaplot::Plot.new
training_data = plt.add(:scatter, x, y)
training_data.title("Training Data")
training_data.color("blue")
linear_regression = plt.add(:line, x, x_by_theta)
linear_regression.title("Linear Regression")
linear_regression.color("red")
plt.legend(true)
plt.show

predict1 = N[[1.0, 3.5], [0.0, 0.0]].dot theta
puts 'For population = 35,000, we predict a profit of ' + (predict1[0]*10000).to_s
predict2 = N[[1.0, 7.0], [0.0, 0.0]].dot theta
puts 'For population = 70,000, we predict a profit of ' + (predict2[0]*10000).to_s

# Part 4: Visualizing J(theta_0, theta_1)
theta0_vals = GSL::Vector.linspace(-10,10,100)
theta1_vals = GSL::Vector.linspace(-1,4,100)

j_vals = NMatrix.new([theta0_vals.length, theta1_vals.length], 0)
for i in 0..(theta0_vals.length-1) do
  for j in 0..(theta1_vals.length-1) do
    t = N[[theta0_vals[i]], [theta1_vals[j]]]
    j_vals[i,j] = compute_cost(x_nmatrix, y_nmatrix, t)
  end
end

x, y = NMatrix.meshgrid([theta0_vals.to_a, theta1_vals.to_a])

# 3-D surface
plt = Nyaplot::Plot3D.new
plt.add(:surface, x.to_a, y.to_a, j_vals.to_a)
plt.show

# contour
plt = Nyaplot::Plot.new
plt.contour(x,Y,j_vals,100)
plt.plot(float(theta[0]),float(theta[1]),'rx')
plt.show
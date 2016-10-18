require 'daru'
require 'nmatrix'
require 'nyaplot'
require 'nyaplot3d'
require 'pp'
require 'gsl'

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
  j = 0
  m = y.shape[0]
  grad = GSL::Matrix.zeros(theta.shape[0], theta.shape[1])

  h = sigmoid_matrix(x * theta)
  j = (1/m) * ((-y.transpose * log_matrix(h)) - ((1-y).transpose * log_matrix(1-h)))
  grad = (1/m) * (x.transpose * (h-y))

  return j, grad
end

def plot_data(x, y)
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

  plt = Nyaplot::Plot.new
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

plot_data(x, y)

m = x.shape[0]
n = x.shape[1]

theta = GSL::Matrix.zeros(3,1)
x_matrix = GSL::Matrix.ones(m, 1).horzcat(x)
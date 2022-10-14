from multiprocessing import Process
import multiprocessing
from time import perf_counter
import numpy

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def alg2(data):
  if len(data) <= 1:
    return data
  else:
    split = len(data) // 2
    left = iter(alg2(data[:split]))
    right = iter(alg2(data[split:]))
    result = []
    left_top = next(left)
    right_top = next(right)
    while True:
      if left_top < right_top:
        result.append(left_top)
        try:
          left_top = next(left)
        except StopIteration:
          return result + [right_top] + list(right)
      else:
        result.append(right_top)
        try:
          right_top = next(right)
        except StopIteration:
          return result + [left_top] + list(left)

if __name__ == '__main__':
    runtime = []
    data = []
    left_t, right_t = 0, 0
    for n in range(2, 8):
        data = data1(10**n)
        start, end = 0, 10**n - 1
        mid = start + (end - start) // 2
        with multiprocessing.Pool(2) as workers:
            start = perf_counter()
            left, right = workers.map(alg2, [data[:mid + 1], data[mid + 1:]])
            while left_t < len(left) and right_t < len(right):
                if left[left_t] <= right[right_t]:
                    data.append(left[left_t])
                    left_t += 1
                else:
                    data.append(right[right_t])
                    right_t += 1
            data += left[left_t:] + right[right_t:]
            end = perf_counter()
            runtime.append(end - start)
    print(runtime)
import tensorflow as tf 

def average(tensor):
    if tf.size(tensor) == 0:
        raise ValueError("Cannot compute the average of an empty tensor.")
    run = tf.reduce_mean(tensor)
    return run


class AverageTest(tf.test.TestCase):
    def test_average(self):
        tensor = tf.constant([1.0, 2.0, 3.0, 4.0])
        result = average(tensor)
        expected = 2.5
        self.assertAllClose(result, expected)

    def test_average_with_empty_tensor(self):
        tensor = tf.constant([])
        with self.assertRaises(ValueError):
            average(tensor)
            
    def test_average_with_multidimensional_tensor(self):
        tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = average(tensor)
        expected = 2.5
        self.assertAllClose(result, expected)

if __name__ == "__main__":
    tf.test.main()
        
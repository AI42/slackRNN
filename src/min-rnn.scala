// load breeze and other dependencies
import scala.Predef
import scala.collection.mutable.ArrayBuffer
import breeze.linalg._
import breeze.numerics._
import scala.io.Source

// read text data
// TODO: read file name from command line argument

class minCharRNN(filename: String) extends RNN {
  // first load input file on instantiation
  var data = Source.fromFile(filename).getLines.mkString
  val dataList = data.toList
  var chars = dataList.toSet.toList


  var data_size = data.length
  var vocab_size = chars.length

  // print loaded stats
  println("Loaded text file has " + data_size + " characters, " + vocab_size + " unique")

  var map_ch_ix = (0 until vocab_size).map((i: Int) => (chars(i) -> i)).toMap
  var map_ix_ch = (0 until vocab_size).map((i: Int) => (i -> chars(i))).toMap

  var xs: ArrayBuffer[DenseMatrix[Double]] = _
  var hs: ArrayBuffer[DenseMatrix[Double]] = _
  var ys: ArrayBuffer[DenseMatrix[Double]] = _
  var ps: ArrayBuffer[DenseMatrix[Double]] = _

  var loss: Double = 0.0

  var dWxh: DenseMatrix[Double] = _
  var dWhh: DenseMatrix[Double] = _
  var dWhy: DenseMatrix[Double] = _
  var dbh: DenseMatrix[Double] = _
  var dby: DenseMatrix[Double] = _
  var dhnext: DenseMatrix[Double] = _

  // set hyperparameters
  val hidden_size = 100
  val history_length = 25
  val momentum = 0.1

  // define empty weight matrices
  var Wxh = DenseMatrix.rand(hidden_size, vocab_size) :* 0.01
  var Whh = DenseMatrix.rand(hidden_size, hidden_size) :* 0.01
  var Why = DenseMatrix.rand(vocab_size, hidden_size) :* 0.01
  var bh = DenseMatrix.zeros[Double](hidden_size, 1)
  var by = DenseMatrix.zeros[Double](vocab_size, 1)

  // setup variables for adagrad iterations
  // counters
  var n = 0; var p = 0

  // memory variables
  var mWxh = DenseMatrix.zeros[Double](hidden_size, vocab_size)
  var mWhh = DenseMatrix.zeros[Double](hidden_size, hidden_size)
  var mWhy = DenseMatrix.zeros[Double](vocab_size, hidden_size)
  var mbh = DenseMatrix.zeros[Double](hidden_size, 1)
  var mby = DenseMatrix.zeros[Double](vocab_size, 1)

  // loss at iteration 0
  var smooth_loss = -log(1.0/vocab_size)*history_length

  var hprev = DenseMatrix.zeros[Double](hidden_size,1)

  // loss function
  def lossFunction(inputs: List[Int], targets: List[Int], hprev: DenseMatrix[Double]): (Double, DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    xs = ArrayBuffer.empty[DenseMatrix[Double]]
    hs = ArrayBuffer.empty[DenseMatrix[Double]]
    ys = ArrayBuffer.empty[DenseMatrix[Double]]
    ps = ArrayBuffer.empty[DenseMatrix[Double]]
    hs += hprev
    var loss = 0.0

    var t = 0

    // forward pass
    for( t <- inputs.indices ) {
      xs += DenseMatrix.zeros[Double](vocab_size, 1)
      xs(t)(inputs(t), 0) = 1
      // hidden state
      val ind = if(t == 0) 0 else t-1
      hs += tanh( Wxh * xs(t) + Whh * hs(ind) + bh )
      ys += Why * hs(t) + by
      ps += exp(ys(t)) / sum(exp(ys(t))) //softmax probability
      loss += -log(ps(t)(targets(t),0)) //cross entropy loss
    }

    // backwards pass
    // gradient matrices
    dWxh = DenseMatrix.zeros[Double](hidden_size, vocab_size)
    dWhh = DenseMatrix.zeros[Double](hidden_size, hidden_size)
    dWhy = DenseMatrix.zeros[Double](vocab_size, hidden_size)
    // for bias units
    dbh = DenseMatrix.zeros[Double](hidden_size, 1)
    dby = DenseMatrix.zeros[Double](vocab_size, 1)

    dhnext = DenseMatrix.zeros[Double](hidden_size,1)
    t = 0
    for( t <- inputs.length-1 to 0 by -1 ) {
      var dy = ps(t).copy
      val tar = targets(t)
      dy(tar,0) -= 1
      dWhy :+= dy * hs(t).t
      dby :+= dy
      var dh = Why.t * dy + dhnext
      var dhraw = (1.0 - (hs(t) :* hs(t))) :* dh
      dbh :+= dhraw
      dWxh :+= dhraw * xs(t).t
      val historystate = if(t == 0) hs.last else hs(t-1)
      dWhh :+= dhraw * historystate.t
      dhnext = Whh.t * dhraw
    }

    // clip gradients
    dWxh = clip(dWxh, -5.0, 5.0)
    dWhh = clip(dWhh, -5.0, 5.0)
    dWhy = clip(dWhy, -5.0, 5.0)
    dbh = clip(dbh, -5.0, 5.0)
    dby = clip(dby, -5.0, 5.0)

    return (loss, dWxh, dWhh, dWhy, dbh, dby, hs(inputs.length - 1))
  }

  // sample function
  def sample(h: DenseMatrix[Double], seed_ix: Int, n: Int): ArrayBuffer[Int] = {
    var x = DenseMatrix.zeros[Double](vocab_size, 1)
    x(seed_ix,0) = 1
    var ixes = ArrayBuffer.empty[Int]
    var hprev = h
    for( t <- 0 until n) {
      hprev = tanh((Wxh * x) + (Whh * hprev) + (bh))
      val y = (Why * hprev) + by
      val p = exp(y) / sum(exp(y))
      // scala/breeze doesn't have the equivalent of np.random.choice
      // need to make the choice differently
      val choice = rand()
      val cdf = accumulate(p.toDenseVector)
      var ix = 0
      (0 until cdf.length).foreach((d: Int) => if(cdf(d) <= choice) ix = d + 1)
      x = DenseMatrix.zeros[Double](vocab_size, 1)
      x(ix,0) = 1
      ixes += ix
    }
    return ixes
  }
  // main function
  def main() = {

    // stochastic gradient / adagrad update
    while(true) {
      // p tracks memory position
      if (p + history_length + 1 >= data_size || n == 0) {
        hprev = DenseMatrix.zeros[Double](hidden_size,1)
        p = 0
      }
      var inputs = data.slice(p, p+history_length).map(map_ch_ix).toList
      var targets = data.slice(p+1, p+history_length+1).map(map_ch_ix).toList

      // sample when the number of iterations is something
      if(n % 10000 == 0) {
        val sample_ix = sample(hprev, inputs.head, 1000)
        val chars = sample_ix.map(map_ix_ch).mkString
        println("---\n")
        println(chars)
        println("\n---")
      }
      // get new gradients
      val gradients = lossFunction(inputs, targets, hprev)
      // unroll the returned Tuple7
      loss = gradients._1; dWxh = gradients._2; dWhh = gradients._3; dWhy = gradients._4; dbh = gradients._5; dby = gradients._6; hprev = gradients._7

      // calculate new loss
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      // print progress
      if(n % 10000 == 0) println("Iteration " + n + ", loss: " + smooth_loss)

      // update parameters
      val weights = List(Wxh, Whh, Why, bh, by)
      val deltas = List(dWxh, dWhh, dWhy, dbh, dby)
      val mems = List(mWxh, mWhh, mWhy, mbh, mby)
      (0 until 5).foreach((i: Int) => {
        mems(i) += deltas(i) :* deltas(i)
        weights(i) += -momentum :* (deltas(i) / sqrt(mems(i) + 1e-8))
      })
      // move counters
      p += history_length
      n += 1

    }
  }

}

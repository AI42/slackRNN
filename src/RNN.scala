import breeze.linalg.DenseMatrix
import scala.collection.mutable.ArrayBuffer
import scala.Predef

trait RNN {
  var data: String
  var chars: List[Char]
  var map_ch_ix: Predef.Map[Char, Int]
  var map_ix_ch: Predef.Map[Int, Char]
  var data_size: Int
  var vocab_size: Int

  var xs: ArrayBuffer[DenseMatrix[Double]]
  var hs: ArrayBuffer[DenseMatrix[Double]]
  var ys: ArrayBuffer[DenseMatrix[Double]]
  var ps: ArrayBuffer[DenseMatrix[Double]]

  var loss: Double

  var dWxh: DenseMatrix[Double]
  var dWhh: DenseMatrix[Double]
  var dWhy: DenseMatrix[Double]
  var dbh: DenseMatrix[Double]
  var dby: DenseMatrix[Double]
  var dhnext: DenseMatrix[Double]

  var Wxh: DenseMatrix[Double]
  var Whh: DenseMatrix[Double]
  var Why: DenseMatrix[Double]
  var bh: DenseMatrix[Double]
  var by: DenseMatrix[Double]

  var mWxh: DenseMatrix[Double]
  var mWhh: DenseMatrix[Double]
  var mWhy: DenseMatrix[Double]
  var mbh: DenseMatrix[Double]
  var mby: DenseMatrix[Double]

  var smooth_loss: Double
  var hprev: DenseMatrix[Double]

}
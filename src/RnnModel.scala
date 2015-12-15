//import statements?

object RnnModel extends App {
  val char_rnn = new minCharRNN("input.txt")
  def main (): Unit = {
    char_rnn.main()

  }
}
import scala.io.Source
import scala.util.parsing.json._

class slackParser(jsonname: String) {
  val rawtext = Source.fromFile(jsonname).getLines().mkString
  val parsed = JSON.parseFull(rawtext)
  val slack_json: List[Any] = parsed match {
      case Some(l: List[Any]) => l
      case None => List()
      case _ => List()
  }
  if (slack_json.length == 0) {
    println("JSON not parsed properly")
  } else {
    println("JSON parsed properly")
  }
}

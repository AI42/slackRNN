import scala.collection.mutable
import scala.io.Source
import scala.util.parsing.json._

class slackParser(jsonname: String) {
  val rawtext = Source.fromFile(jsonname).getLines().mkString
  val parsed = JSON.parseFull(rawtext)
  val slack_json: List[Map[String, String]] = parsed match {
      case Some(l: List[Map[String, String]]) => l
      case None => List()
      case _ => List()
  }
  def main(): Option[collection.mutable.Map[String, String]] = {
    if (slack_json.length == 0) {
      println("JSON not parsed properly")
      None
    } else {
      println("JSON parsed properly")
      // first remove all subtype messages nad only keep messages
      val messages = slack_json.filter((m: Map[String, String]) => m.head._1 == "type")
      // then get all usernames
      val users = slack_json.map((m: Map[String, String]) => m("user")).toSet.toList
      // then for each user, match messages to user and return a user -> text map
      // prepare mutable map
      var user_map = collection.mutable.Map[String, String]()
      users.foreach((u: String) => user_map += (u -> " "))
      // save text to map
      messages.foreach((m: Map[String, String]) => user_map(m("user")) += "\n\n" + m("text"))
      println("parsed")
      Some(user_map) // return map
    }
  }

}

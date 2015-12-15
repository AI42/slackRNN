import sun.font.TrueTypeFont

import scala.collection.mutable
import scala.io.Source
import scala.util.parsing.json._
import java.io._

class slackParser(folder: String, user_json: String) {
  // keep user json file outside the main archive folder

  val json_format = """.*\.json$""".r

  def singleFile(jsonname: String): Option[collection.mutable.Map[String, String]] = {
    // get raw text from JSON file
    val rawtext = Source.fromFile(jsonname).getLines().mkString
    // parse raw text as JSON - returns option of List of Maps
    val parsed = JSON.parseFull(rawtext)
    // match option with value
    val slack_json: List[Map[String, String]] = parsed match {
      case Some(l: List[Map[String, String]]) => l
      case None => List()
      case _ => List()
    }
    // check if I parsed something
    if (slack_json.length == 0) {
      // the JSON file failed to parse, return None type
      println("JSON not parsed properly")
      None
    } else {
      println("JSON parsed properly")
      // first remove all subtype messages nad only keep messages
      val messages = slack_json.filter((m: Map[String, String]) => m.head._1 == "type")
      // then get all usernames
      val users = messages.map((m: Map[String, String]) => m("user")).toSet.toList
      // then for each user, match messages to user and return a user -> text map
      // prepare mutable map
      var user_map = collection.mutable.Map[String, String]()
      users.foreach((u: String) => user_map += (u -> " "))
      // save text to map
      messages.foreach((m: Map[String, String]) => user_map(m("user")) += "\n\n" + m("text"))
      // return Map as an Option
      Some(user_map)
    }
  }

  def isJsonFile(f: File): Boolean = {
    val name = f.getName
    name match {
        case json_format(_*) => true
        case _ => false
    }
  }

  def getJsonFiles(path: File): Array[File] = {
    val all_files = path.listFiles
    all_files.filter(isJsonFile) ++ all_files.filter(_.isDirectory).flatMap(getJsonFiles)
  }

  def mapMerger(master: mutable.Map[String, String], branch: mutable.Map[String, String]) = {
    branch.filter( (t: (String, String) ) => master.keySet contains t._1 ).foreach((s: (String, String)) => master(s._1) += "\n\n" + s._2)
    branch.filterNot( (t: (String, String) ) => master.keySet contains t._1 ).foreach((s: (String, String)) => master += s)
  }

  def parseUsers(userfile: String): Map[String, String] = {
    val rawtext = Source.fromFile(userfile).getLines().mkString
    // parse raw text as JSON - returns option of List of Maps
    val parsed = JSON.parseFull(rawtext).get
    val id_name = parsed.map((m: Map[String, String]) => (m("id") -> m("name"))).toMap
    id_name
  }
  def main() = {
    // get all json filenames from the given folder (recursively too)
    val fold = new File(folder)
    val all_json_files = getJsonFiles(fold)
    // create "master map" to hold all text
    var master_map = collection.mutable.Map[String, String]()
    // parse each file, get maps from them
    all_json_files.foreach((f: File) => singleFile(f.getPath) match {
        case Some(m: mutable.Map[String, String]) => mapMerger(master_map, m)
        case None => Unit
    })
    // TODO: fetch all user names, link IDs with usernames and replace them in output and filenames
    // fetch all users and get a map of ID -> username

    // write maps as txt files, one for each user
    master_map.foreach((u: (String, String)) => {
      val filename = u._1 + ".txt"
      val newfile = new File(filename)
      newfile.createNewFile()
      val pw = new PrintWriter(newfile)
      pw.write(u._2)
      pw.close
    })
  }

}

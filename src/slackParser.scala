import sun.font.TrueTypeFont

import scala.collection.mutable
import scala.io.Source
import scala.util.parsing.json._
import java.io._

class slackParser(folder: String, user_json: String, channel_json: String) {
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
      // first remove all subtype messages nad only keep messages
      val messages = slack_json.filter((m: Map[String, String]) => m.head._1 == "type")
      // then get all usernames
      val users = messages.map((m: Map[String, String]) => m("user")).toSet.toList
      // then for each user, match messages to user and return a user -> text map
      // prepare mutable map
      var user_map = collection.mutable.Map[String, String]()
      users.foreach((u: String) => if (u != "USLACKBOT") user_map += (u -> " "))
      // save text to map
      messages.foreach((m: Map[String, String]) => if (m("user") != "USLACKBOT") user_map(m("user")) += "\n\n" + m("text"))
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

  // parse list of users with their IDs
  def parseUsers(userfile: String): Map[String, String] = {
    val rawtext = Source.fromFile(userfile).getLines().mkString
    // parse raw text as JSON - returns option of List of Maps
    val parsed: List[Map[String,String]] = JSON.parseFull(rawtext) match {
      case Some(l: List[Map[String, String]]) => l
      case None => List()
      case _ => List()
    }
    val id_name = parsed.map((m: Map[String, String]) => m("id") -> m("name")).toMap
    id_name
  }

  // parse list of channels with their IDs
  def parseChannels(channelfile: String): Map[String, String] = {
    val rawtext = Source.fromFile(channelfile).getLines().mkString
    // parse raw text as JSON - returns option of List of Maps
    val parsed: List[Map[String,String]] = JSON.parseFull(rawtext) match {
      case Some(l: List[Map[String, String]]) => l
      case None => List()
      case _ => List()
    }
    val id_name = parsed.map((m: Map[String, String]) => m("id") -> m("name")).toMap
    id_name
  }

  // clean a given string of text
  def clean(users: Map[String, String], channels: Map[String, String], text: String): String = {
    var newtext = text
    users.foreach((u: (String, String)) => newtext = newtext.replaceAll("<@" + u._1 + ">", "@"+u._2)) // clean user IDs
    channels.foreach((c: (String, String)) => newtext = newtext.replaceAll("<#" + c._1 + ">", "#"+c._2)) // clean channel IDs
    // replace @channel, @everyone
    newtext = newtext.replaceAll("<!channel>", "@channel")
    newtext = newtext.replaceAll("<!everyone", "@everyone")
    // replace safe/escaped characters, especially ">"
    newtext = newtext.replaceAll("&gt;", ">")
    newtext
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
    // fetch all users and get a map of ID -> username
    val usernames = parseUsers(user_json)
    // fetch all channels and get a map of ID -> channel name
    val channels = parseChannels(channel_json)
    // replace usernames and channel names
    master_map = master_map.map((m: (String, String)) => usernames(m._1) -> clean(usernames, channels, m._2))

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

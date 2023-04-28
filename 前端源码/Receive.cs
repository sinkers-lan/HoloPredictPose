using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.Networking;

public class Receive : MonoBehaviour
{

    public string remoteHost = "123.56.44.128:8001";
    public string mode = "pred";
    public double fps = 5;
    private int counter;
    private GameObject[] offs;
    private player_scripts[] pss;


    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(Get("start"));
        offs = GameObject.FindGameObjectsWithTag("GameController");
        if(offs.Length == 0)
        {
            Debug.LogError("can't find obj with tag.");
        }
        else if (offs.Length == 2)
        {
            //Debug.Log(offs[0].name);
            //Debug.Log(offs[1].name);
            Debug.Log("find obj with tag.");
            pss = new player_scripts[2];
            pss[0] = offs[0].GetComponent<player_scripts>();
            pss[1] = offs[1].GetComponent<player_scripts>();
            //Debug.Log(pss[0].mode + pss[1].mode);
        }
    }
    
    // Update is called once per frame
    void Update()
    {

    }

    private void FixedUpdate()
    {
        double max_counter = ((1.0 / fps) / 0.02);
        //Debug.Log("max counter:" + max_counter);
        counter++;
        counter %= (int)max_counter;
        if (counter == 0)
        {
            StartCoroutine(Get("cv"));
        }
    }

    private void OnApplicationQuit()
    {
        StartCoroutine(Get("end"));
    }

    IEnumerator Get(string arg)
    {
        using (UnityWebRequest webRequest = new UnityWebRequest())
        {
            webRequest.url = "http://" + remoteHost + "/" + arg;
            webRequest.method = UnityWebRequest.kHttpVerbGET;
            webRequest.downloadHandler = new DownloadHandlerBuffer();
            yield return webRequest.SendWebRequest();
            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(webRequest.error);
            }
            else
            {
                
                if(arg == "cv")
                {
                    //Debug.Log(webRequest.downloadHandler.text);
                    //MyJson myJson = JsonUtility.FromJson<MyJson>(webRequest.downloadHandler.text);
                    MyJson myJson = JsonConvert.DeserializeObject<MyJson>(webRequest.downloadHandler.text);
                    //JsonConvert
                    Debug.Log(myJson.avail);
                    Debug.Log(myJson.org.Count);
                    if (myJson.avail == 1)
                    {
                        //Debug.Log(22222);

                        //Debug.Log("pred:" + myJson.pred);
                        pss[0].player(myJson.pred);
                        //Debug.Log("org:" + myJson.pred);
                        pss[1].player(myJson.org);
                        
                        
                        
                    }
                    
                }
            }
        }
    }
    [System.Serializable]
    public class MyJson
    {
        public List<List<double>> org { get; set; }
        public List<List<double>> pred { get; set; }
        public int avail { get; set; }
    }
}
 
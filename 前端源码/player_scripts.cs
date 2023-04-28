using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Runtime.Serialization.Json;
using System.IO;
using System;
using UnityEngine.Animations.Rigging;
using System.Threading;

public class player_scripts : MonoBehaviour
{
    // Start is called before the first frame update
    public static player_scripts Instance_org;
    public static player_scripts Instance_pred;
    public static player_scripts Instance;
    public string mode = "pred";
    public string jsonfile;
    public bool useLocalJson = true;
    List<GameObject> offsets = new List<GameObject>();
    List<List<List<double>>> motionList;
    int motionI = 0;
    //Dictionary<string, Vector3> orgVec = new Dictionary<string, Vector3>();
    Dictionary<string, Quaternion> orgQua = new Dictionary<string, Quaternion>();
    List<List<double>> lastFrameForJoints = null;
    Quaternion leftFootOrgQua, rightFootOrgQua;
    double offsetWeight = 1;
    private void Awake()
    {
        if(mode == "pred")
        {
            Instance_pred = this;
        }
        else
        {
            Instance_org = this;
        }
        Instance = this;
        
    }

    //[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)] 
    //static void Init() 
    //{
    //    Debug.Log("Instance reset.");
    //    Instance_org = null;
    //    Instance_pred = null;
    //}

    void Start()
    {
        GameObject offsetObj = this.gameObject;
        getAllChild(offsetObj);
        //foreach (GameObject bone in offsets)
        //{
        //        orgVec.Add(bone.name, bone.transform.eulerAngles);
        //    //Debug.Log(bone.name + " " + bone.transform.eulerAngles.x + ", " + bone.transform.eulerAngles.y + ", " + bone.transform.eulerAngles.z);
        //    //Debug.Log(int.Parse(bone.name.Split('_')[bone.name.Split('_').Length - 1]));
        //}

        getInfo();
        
    }

void getAllChild(GameObject father)
    {
        OverrideTransform ot = father.GetComponent(typeof(OverrideTransform)) as OverrideTransform;
        if (ot)
        {
            //Debug.Log(ot.weight);
            //Debug.Log(father.name + ot.data.constrainedObject.gameObject.transform.eulerAngles);
            //orgVec.Add(father.name, ot.data.constrainedObject.gameObject.transform.eulerAngles);
            orgQua.Add(father.name, ot.data.constrainedObject.gameObject.transform.rotation);
            if (ot.name == "rightFoot_offset")
            {
                rightFootOrgQua = ot.data.constrainedObject.gameObject.transform.localRotation;
            }
            if (ot.name == "leftFoot_offset")
            {
                leftFootOrgQua = ot.data.constrainedObject.gameObject.transform.localRotation;
            }
        }
        else
        {
            //Debug.Log("can't get component");
        }
        //Debug.Log("Parent:" + father.name);
        foreach (Transform child in father.transform)
        {
            //Debug.Log("Child:" + child.gameObject.name);
            if(child.name != "foot")
            {
                offsets.Add(child.gameObject);
            }
            
            getAllChild(child.gameObject);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void Reset()
    {
        foreach (GameObject bone in offsets)
        {
            //Debug.Log("reset " + bone.name);
            bone.transform.localEulerAngles = new Vector3(0, 0, 0);
        }
        Debug.Log("Done reset.");
    }

    private void FixedUpdate()
    {
        if (useLocalJson)
        {
            if (motionI < motionList.Count - 1)
            {
                player(motionList[motionI]);
                motionI++;
            }
            else
            {
                offsetWeight -= 0.02;
                playerWait((float)offsetWeight);
            }
        }
        
    }

    public void player(List<List<double>> aFrameForJoints)
    {
        Debug.Log("enter.");
        // һ��17���ؼ��㣬��aFrameForJoints��17�У�offsets��16������offsets�Ĺؼ���˳���17����Ӧ
        //for (int i = 0; i < aFrameForJoints.Count; i++)
        //{
        //    GameObject bone = offsets[i];
        //    List<double> coor = aFrameForJoints[i];
        //    double x = coor[0];
        //    double y = coor[1];
        //    double z = coor[2];
        //}
        foreach (GameObject bone in offsets)
        {
            

            Dictionary<int, int> transIndex = new Dictionary<int, int>();
            transIndex.Add(0, 0);
            transIndex.Add(1, 1);
            transIndex.Add(2, 2);
            transIndex.Add(3, 3);
            transIndex.Add(6, 4);
            transIndex.Add(7, 5);
            transIndex.Add(8, 6);
            transIndex.Add(12, 7);
            transIndex.Add(13, 8);
            transIndex.Add(14, 9);
            transIndex.Add(15, 10);
            transIndex.Add(17, 11);
            transIndex.Add(18, 12);
            transIndex.Add(19, 13);
            transIndex.Add(25, 14);
            transIndex.Add(26, 15);
            transIndex.Add(27, 16);
            
            double x, y, z;
            double x1, y1, z1;
            double x2, y2, z2;
            if (lastFrameForJoints == null)
            {

            }
            //Debug.Log("start Split." + bone.name);
            switch (bone.name.Split('_')[0])
            {
                case "hips":

                    // ת���Ƕȵ�������
                    List<double> j0 = aFrameForJoints[transIndex[0]];
                    List<double> j6 = aFrameForJoints[transIndex[6]];
                    x = j0[0];
                    y = j0[0];
                    z = j0[0];
                    x1 = j0[0]-1;
                    y1 = j0[0];
                    z1 = j0[0];
                    x2 = j6[0];
                    y2 = j6[1];
                    z2 = j6[2];
                    break;
                case "spine":
                    j0 = aFrameForJoints[transIndex[0]];
                    List<double> j12 = aFrameForJoints[transIndex[12]];
                    x = j0[0];
                    y = j0[0];
                    z = j0[0];
                    x1 = j0[0];
                    y1 = j0[0]+1;
                    z1 = j0[0];
                    x2 = j12[0];
                    y2 = j12[1];
                    z2 = j12[2];
                    break;
                case "chest":
                    j12 = aFrameForJoints[transIndex[12]];
                    List<double> j13 = aFrameForJoints[transIndex[13]];
                    x = j12[0];
                    y = j12[1];
                    z = j12[2];
                    x1 = j12[0];
                    y1 = j12[1] + 1;
                    z1 = j12[2];
                    x2 = j13[0];
                    y2 = j13[1];
                    z2 = j13[2];
                    break;
                case "neck":
                    j13 = aFrameForJoints[transIndex[13]];
                    List<double> j15 = aFrameForJoints[transIndex[15]];
                    x = j13[0];
                    y = j13[1];
                    z = j13[2];
                    x1 = j13[0];
                    y1 = j13[1] + 1;
                    z1 = j13[2];
                    x2 = j15[0];
                    y2 = j15[1];
                    z2 = j15[2];
                    break;
                case "leftShoulder":
                    j13 = aFrameForJoints[transIndex[13]];
                    List<double> j17 = aFrameForJoints[transIndex[17]];
                    x = j13[0];
                    y = j13[1];
                    z = j13[2];
                    x1 = j13[0] - 1;
                    y1 = j13[1];
                    z1 = j13[2];
                    x2 = j17[0];
                    y2 = j17[1];
                    z2 = j17[2];
                    break;
                case "leftUpperArm":
                    j17 = aFrameForJoints[transIndex[17]];
                    List<double> j18 = aFrameForJoints[transIndex[18]];
                    x = j17[0];
                    y = j17[1];
                    z = j17[2];
                    x1 = j17[0] - 1;
                    y1 = j17[1];
                    z1 = j17[2];
                    x2 = j18[0];
                    y2 = j18[1];
                    z2 = j18[2];
                    break;
                case "leftLowerArm":
                    j18 = aFrameForJoints[transIndex[18]];
                    List<double> j19 = aFrameForJoints[transIndex[19]];
                    x = j18[0];
                    y = j18[1];
                    z = j18[2];
                    x1 = j18[0] - 1;
                    y1 = j18[1];
                    z1 = j18[2];
                    x2 = j19[0];
                    y2 = j19[1];
                    z2 = j19[2];
                    break;
                case "rightShoulder":
                    j13 = aFrameForJoints[transIndex[13]];
                    List<double> j25 = aFrameForJoints[transIndex[25]];
                    x = j13[0];
                    y = j13[1];
                    z = j13[2];
                    x1 = j13[0] + 1;
                    y1 = j13[1];
                    z1 = j13[2];
                    x2 = j25[0];
                    y2 = j25[1];
                    z2 = j25[2];
                    break;
                case "rightUpperArm":
                    j25 = aFrameForJoints[transIndex[25]];
                    List<double> j26 = aFrameForJoints[transIndex[26]];
                    x = j25[0];
                    y = j25[1];
                    z = j25[2];
                    x1 = j25[0] + 1;
                    y1 = j25[1];
                    z1 = j25[2];
                    x2 = j26[0];
                    y2 = j26[1];
                    z2 = j26[2];
                    break;
                case "rightLowerArm":
                    j26 = aFrameForJoints[transIndex[26]];
                    //Debug.Log(transIndex[27]);
                    //Debug.Log(aFrameForJoints.Count);
                    List<double> j27 = aFrameForJoints[transIndex[27]];
                    x = j26[0];
                    y = j26[1];
                    z = j26[2];
                    x1 = j26[0] + 1;
                    y1 = j26[1];
                    z1 = j26[2];
                    x2 = j27[0];
                    y2 = j27[1];
                    z2 = j27[2];
                    break;
                case "leftUpperLeg":
                    j6 = aFrameForJoints[transIndex[6]];
                    List<double> j7 = aFrameForJoints[transIndex[7]];
                    x = j6[0];
                    y = j6[1];
                    z = j6[2];
                    x1 = j6[0];
                    y1 = j6[1] - 1;
                    z1 = j6[2];
                    x2 = j7[0];
                    y2 = j7[1];
                    z2 = j7[2];
                    break;
                case "leftLowerLeg":
                    j7 = aFrameForJoints[transIndex[7]];
                    List<double> j8 = aFrameForJoints[transIndex[8]];
                    x = j7[0];
                    y = j7[1];
                    z = j7[2];
                    x1 = j7[0];
                    y1 = j7[1] - 1;
                    z1 = j7[2];
                    x2 = j8[0];
                    y2 = j8[1];
                    z2 = j8[2];
                    break;
                case "rightUpperLeg":
                    List<double> j1 = aFrameForJoints[transIndex[1]];
                    List<double> j2 = aFrameForJoints[transIndex[2]];
                    x = j1[0];
                    y = j1[1];
                    z = j1[2];
                    x1 = j1[0];
                    y1 = j1[1] - 1;
                    z1 = j1[2];
                    x2 = j2[0];
                    y2 = j2[1];
                    z2 = j2[2];

                    break;
                case "rightLowerLeg":
                    j2 = aFrameForJoints[transIndex[2]];
                    List<double> j3 = aFrameForJoints[transIndex[3]];
                    x = j2[0];
                    y = j2[1];
                    z = j2[2];
                    x1 = j2[0];
                    y1 = j2[1] - 1;
                    z1 = j2[2];
                    x2 = j3[0];
                    y2 = j3[1];
                    z2 = j3[2];

                    //offset.transform.position = Vector3.Lerp(offset.transform.position, new Vector3(1, 1, 1), 0.001f);
                    break;

                default:
                    continue;
            }

            double angle = Angle3(new Point(x, y, z), new Point(x1, y1, z1), new Point(x2, y2, z2));
            //double[] vec = FaXiangLiang(x, y, z, x1, y1, z1, x2, y2, z2);
            Vector3 fxl = Vector3.Cross(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)));
            //Quaternion q = Quaternion.AngleAxis((float)angle, new Vector3((float)vec[0], (float)vec[1], (float)vec[2]));
            Quaternion q = Quaternion.AngleAxis((float)angle, fxl);


            double third_angle;

            Quaternion orgAngle2 = orgQua[bone.name];

            if (bone.name.Split('_')[0] == "spine" || bone.name.Split('_')[0] == "chest" || bone.name.Split('_')[0] == "neck")
            {
                List<double> j1 = aFrameForJoints[transIndex[1]];
                x = 0;
                y = 0;
                z = 0;
                x1 = 1;
                y1 = 0;
                z1 = 0;
                x2 = j1[0];
                y2 = 0;
                z2 = j1[2];
                double hips_angle = Angle3(new Point(x, y, z), new Point(x1, y1, z1), new Point(x2, y2, z2));
                Vector3 hips_fxl = Vector3.Cross(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)));

                List<double> j17 = aFrameForJoints[transIndex[17]];
                List<double> j25 = aFrameForJoints[transIndex[25]];
                x = j17[0];
                y = 0;
                z = j17[2];
                x1 = j17[0] + 1;
                y1 = 0;
                z1 = j17[2];
                x2 = j25[0];
                y2 = 0;
                z2 = j25[2];
                double shoulder_angle = Angle3(new Point(x, y, z), new Point(x1, y1, z1), new Point(x2, y2, z2));
                Vector3 shoulder_fxl = Vector3.Cross(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)));

                if (bone.name.Split('_')[0] == "neck")
                {
                    //Quaternion q_y = Quaternion.AngleAxis((float)shoulder_angle, shoulder_fxl);
                    //bone.transform.rotation = q * (q_y * orgAngle2);
                    //continue;
                    List<double> j13 = aFrameForJoints[transIndex[13]];
                    List<double> j14 = aFrameForJoints[transIndex[14]];
                    x = j13[0];
                    y = 0;
                    z = j13[2];
                    x1 = j13[0];
                    y1 = 0;
                    z1 = j13[2] + 1;
                    x2 = j14[0];
                    y2 = 0;
                    z2 = j14[2];
                    third_angle = Angle3(new Point(x, y, z), new Point(x1, y1, z1), new Point(x2, y2, z2));
                    Vector3 third_fxl = Vector3.Cross(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)));
                    Quaternion third_q = Quaternion.AngleAxis((float)third_angle, third_fxl);
                    bone.transform.rotation = q * (third_q * orgAngle2);
                    continue;
                }

                bool hips_face, shoulder_face;
                if (hips_fxl.y > 0) hips_face = true;
                else hips_face = false;
                if (shoulder_fxl.y > 0) shoulder_face = true;
                else shoulder_face = false;
                if (hips_face != shoulder_face)
                {
                    //Debug.Log("hips_face:( " + hips_fxl.x + ", " + hips_fxl.y + ", " + hips_fxl.z + ")");
                    //Debug.Log("shoulder_face:( " + shoulder_fxl.x + ", " + shoulder_fxl.y + ", " + shoulder_fxl.z + ")");
                    shoulder_angle = -shoulder_angle;
                }

                if (bone.name.Split('_')[0] == "spine")
                {

                    double spine_angle = (hips_angle * (2.0 / 3.0)) + (shoulder_angle * (1.0 / 3.0));
                    //Debug.Log("hips_angle:" + hips_angle);
                    //Debug.Log("shoulder_angle:" + shoulder_angle);
                    //Debug.Log("spine_angle:" + spine_angle);
                    Quaternion q_y = Quaternion.AngleAxis((float)spine_angle, hips_fxl);
                    bone.transform.rotation = q * (q_y * orgAngle2);
                    continue;
                }
                if (bone.name.Split('_')[0] == "chest")
                {
                    double chest_angle = 1.0 / 3.0 * hips_angle + 2.0 / 3.0 * shoulder_angle;
                    //Debug.Log("chest_angle:" + chest_angle);
                    Quaternion q_y = Quaternion.AngleAxis((float)chest_angle, hips_fxl);
                    bone.transform.rotation = q * (q_y * orgAngle2);
                    continue;
                }
            }

            if (bone.name.Split('_')[0] == "hips")
            {
                List<double> j13 = aFrameForJoints[transIndex[13]];
                x = 0;
                y = 0;
                z = 0;
                x1 = 0;
                y1 = 1;
                z1 = 0;
                x2 = 0;
                y2 = j13[1];
                z2 = j13[2];
                third_angle = Angle3(new Point(x, y, z), new Point(x1, y1, z1), new Point(x2, y2, z2));
                Vector3 third_fxl = Vector3.Cross(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)));
                Quaternion third_q = Quaternion.AngleAxis((float)third_angle, third_fxl);
                bone.transform.rotation = q * (third_q * orgAngle2);
                continue;
            }
            if (bone.name.Split('_')[0] == "leftLowerLeg" || bone.name.Split('_')[0] == "rightLowerLeg")
            {
                Vector3 fa_xian;
                if (bone.name.Split('_')[0] == "leftLowerLeg")
                {
                    List<double> j6 = aFrameForJoints[transIndex[6]];
                    List<double> j7 = aFrameForJoints[transIndex[7]];
                    List<double> j8 = aFrameForJoints[transIndex[8]];

                    x = j8[0];
                    y = j8[1];
                    z = j8[2];

                    x1 = j8[0];
                    y1 = j8[1];
                    z1 = j8[2] + 1;

                    x2 = j7[0];
                    y2 = j7[1];
                    z2 = j7[2];
                    fa_xian = new Vector3((float)(j8[0] - j6[0]), (float)(j8[1] - j6[1]), (float)(j8[2] - j6[2]));
                    
                }
                else
                {
                    List<double> j1 = aFrameForJoints[transIndex[1]];
                    List<double> j2 = aFrameForJoints[transIndex[2]];
                    List<double> j3 = aFrameForJoints[transIndex[3]];

                    x = j3[0];
                    y = j3[2];
                    z = j3[2];

                    x1 = j3[0];
                    y1 = j3[2];
                    z1 = j3[2] + 1;

                    x2 = j2[0];
                    y2 = j2[2];
                    z2 = j2[2];
                    fa_xian = new Vector3((float)(j3[0] - j1[0]), (float)(j3[1] - j1[1]), (float)(j3[2] - j1[2]));
                }

                Vector3 v1 = Vector3.ProjectOnPlane(new Vector3((float)(x1 - x), (float)(y1 - y), (float)(z1 - z)), fa_xian).normalized;
                Vector3 v2 = Vector3.ProjectOnPlane(new Vector3((float)(x2 - x), (float)(y2 - y), (float)(z2 - z)), fa_xian).normalized;
                third_angle = Angle3(new Point(x, y, z), new Point(v1.x + x, v1.y + y, v1.z + z), new Point(v2.x + x, v2.y + y, v2.z + z));
                Vector3 fa_xiang_liang = Vector3.Cross(v1, v2).normalized;


                GameObject foot, parent;
                parent = bone.transform.parent.Find("foot").gameObject;
                if (bone.name.Split('_')[0] == "leftLowerLeg") 
                    foot = parent.transform.Find("leftFoot_offset").gameObject;
                else
                    foot = parent.transform.Find("rightFoot_offset").gameObject;
                if (foot)
                {
                    int face;
                    if (fa_xiang_liang.y > 0) face = 1;
                    else face = -1;
                    Quaternion third_q = Quaternion.AngleAxis((float)third_angle, new Vector3(face, 0, 0));
                    if (bone.name.Split('_')[0] == "leftLowerLeg")
                        foot.transform.localRotation = third_q * leftFootOrgQua;
                    else
                        foot.transform.localRotation = third_q * rightFootOrgQua;
                }
                else
                {
                    Debug.Log("foot not found.");
                }
                bone.transform.rotation = q * orgAngle2;
                continue;
            }

            bone.transform.rotation = q * orgAngle2;
        }
    }

    void playerWait(float w)
    {
        foreach (GameObject obj in offsets)
        {
            OverrideTransform ot = obj.GetComponent(typeof(OverrideTransform)) as OverrideTransform;
            if (ot)
            {
                ot.weight = w;
            }
        }
    }

     void getInfo()
    {
        
        string jsonString = File.ReadAllText(jsonfile);
        
        motionList = JsonToObject<List<List<List<double>>>>(jsonString, System.Text.Encoding.UTF8);
        
    }

    T JsonToObject<T>(string json, System.Text.Encoding encoding)
    {
        T resultObject = System.Activator.CreateInstance<T>();
        DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(T));
        using (System.IO.MemoryStream ms = new System.IO.MemoryStream(encoding.GetBytes(json)))
        {
            resultObject = (T)serializer.ReadObject(ms);
        }
        return resultObject;
    }


    double Angle3(Point cen, Point first, Point second)
    {
        double dx1, dx2, dy1, dy2, dz1, dz2;
        double angle;

        dx1 = first.X - cen.X;
        dy1 = first.Y - cen.Y;
        dz1 = first.Z - cen.Z;

        dx2 = second.X - cen.X;
        dy2 = second.Y - cen.Y;
        dz2 = second.Z - cen.Z;

        double c = (double)Math.Sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1) * (double)Math.Sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        if (c == 0) return 0;

        angle = (double)Math.Acos((dx1 * dx2 + dy1 * dy2 + dz1 * dz2) / c) * 180 / Math.PI;

        return angle;
    }

}

public class Point
{
    public double X;
    public double Y;
    public double Z;

    public Point(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }
}
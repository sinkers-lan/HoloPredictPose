using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class StreamerHL2 : MonoBehaviour
{
    // Start is called before the first frame update
    // [DllImport("HL2RmStreamUnityPlugin", EntryPoint = "StartStreaming", CallingConvention = CallingConvention.StdCall)];
    //[DllImport("HL2RmStreamUnityPlugin", EntryPoint = "StartStreaming", CallingConvention = CallingConvention.StdCall)]
    //private static extern void StartDll(); 这是旧的函数入口点。更新为下面 2021-11-1
    [DllImport("HL2RmStreamUnityPlugin", EntryPoint = "Initialize", CallingConvention = CallingConvention.StdCall)]
    public static extern void InitializeDll(); // 更新的函数入口点  2021-11-1

    void Start()
    {

        //StartDll();
        InitializeDll();

    }

    // Update is called once per frame
    void Update()
    {

    }
}


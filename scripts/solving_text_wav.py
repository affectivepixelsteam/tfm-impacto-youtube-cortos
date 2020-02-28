import os
import shutil

files =[
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/6lreklhpn_A.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/6YfmDvGHkWI.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/7F9Nkf-brY8.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/9CmEvD-KSF4.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/725l-lxdaZ8.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/BB7EUzwp3tY.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/c4XNoFmaVME.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/Cxiibyb78p4.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/EoLTaTl8K1I.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/evdorzmM0As.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/i7uMVZ6d68w.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/I7ueMy6xhOQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/INirgL-t0y8.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/j2Cn2t602wg.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/KkY6c6nnXb4.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/M8UW3i01S3s.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/oKhXvJbPr5M.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/OkIbiuBMfHg.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/On1kAuLCUUw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/opewszhqosw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/pcBJy0afdRM.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/PSzqeEACG6M.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/Q3FplDAxHlI.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/qFMNYoZfUTA.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/RWK9ckz2TRE.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/ujaI_TlYVYA.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/ulZgJMBw1Dc.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/WBP9nfjh5TM.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/y2QWnGLvGhI.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2015/zWLaJklZU5k.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/0gfwywKarnQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/2OxmwFP_A6E.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/5SpOEepBwUw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/09mw9ZKBK_s.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/9t9jFFpj9c8.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/Asjfo8Wtj44.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/BHHqxN6HBG0.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/CeuY0OA7DxQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/EnhInwHIUrA.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/FhawcZgS_sI.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/H1wtUs-lYh8.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/IbeET4Tw45w.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/IGobPyfVa6o.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/IVmSinMQBNc.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/jWXJPMF10_o.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/Kxfwul4x84M.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/lbLvtNwi0sw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/Ljyu3qMoUcM.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/m5ww528wpMQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/mt6uUsWc06o.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ns7jFEdKWXY.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/OiXlGMh5Qqk.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ojGNJNYRfuE.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/pCXvcEsE_EE.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/s-3QbtAEx8k.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/S8EDksE0Fu4.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/SdM1JdCddSw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/SE_SBYKdGFQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/TMfRN5nfMSk.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ucIB3wQROYc.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/xuoTofVXK_A.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/Y5HSmqIfqgM.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/yX0Ptdlj504.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ZJabcsYDEpw.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/znGapr3Kgqg.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ZqlBOtSWarQ.wav',
    '/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas/2018/ZQPYSWhw7sQ.wav']

videoid = []

for f in files:
    name = f[55:66]
    videoid.append(name)
    os.remove(f)

years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        if directorio in videoid:
            path_inside_directorio = os.path.join(folder_path, directorio)
            os.chdir(path_inside_directorio)
            path_to_audio = directorio + '.wav'
            os.remove(path_to_audio)
            video = directorio + '.mp4'
            cmd = 'ffmpeg -i ' + video + ' ' + directorio + '.wav'
            os.system(cmd)
from moviepy.editor import VideoFileClip, clips_array

def concat_side_by_side(video1_path, video2_path, output_path="output_side_by_side_new_fusion.mp4"):
    # Load video clips
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    # Resize to same height (so they align correctly)
    if clip1.h != clip2.h:
        clip2 = clip2.resize(height=clip1.h)

    # Stack side by side
    final_clip = clips_array([[clip1, clip2]])

    # Export result
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Close to free memory
    clip1.close()
    clip2.close()
    final_clip.close()

if __name__ == "__main__":
    # concat_side_by_side("/home/student/hwu/Workplace/ASLDataset_Annotation/dataset/clip/3h7veASAUaw-046.mp4", "/home/student/hwu/Workplace/AiOS/demo/demo_asl/3h7veASAUaw-046/3h7veASAUaw-046_demo.mp4", "merged_mp4s/3h7veASAUaw-046.mp4")
    # concat_side_by_side("/home/student/hwu/Dataset/how2sign/raw_videos/G1hb5HugzVk_4-8-rgb_front.mp4", "/home/student/hwu/Workplace/AiOS/demo/demo_how2sign/G1hb5HugzVk_4-8-rgb_front/G1hb5HugzVk_4-8-rgb_front_demo.mp4", "merged_mp4s/G1hb5HugzVk_4-8.mp4")
    # concat_side_by_side("/home/student/hwu/Workplace/AiOS/_fZbAxSSbX4_18-5-rgb_front_talkshow/_fZbAxSSbX4_18-5-rgb_front.mp4","/home/student/hwu/Workplace/AiOS/_fZbAxSSbX4_18-5-rgb_front/_fZbAxSSbX4_18-5-rgb_front.mp4")
    # concat_side_by_side("-g0iPSnQt6w_6-1-rgb_front/-g0iPSnQt6w_6-1-rgb_front.mp4","/data/hwu/how2sign/raw_videos_test/-g0iPSnQt6w_6-1-rgb_front.mp4") #text+audio-only
    # concat_side_by_side("_0-JkwZ9o4Q_5-5-rgb_front/_0-JkwZ9o4Q_5-5-rgb_front.mp4","/data/hwu/how2sign/raw_videos_train/_0-JkwZ9o4Q_5-5-rgb_front.mp4") #text+audio fuison(new)
    # concat_side_by_side("f8ShD9YwEfo_18-2-rgb_front/f8ShD9YwEfo_18-2-rgb_front.mp4","/data/hwu/how2sign/raw_videos_train/f8ShD9YwEfo_18-2-rgb_front.mp4") #text+audio fuison(new)
    # concat_side_by_side("_fZbAxSSbX4_0-5-rgb_front_origin/_fZbAxSSbX4_0-5-rgb_front.mp4","_fZbAxSSbX4_0-5-rgb_front_new/_fZbAxSSbX4_0-5-rgb_front.mp4") #text+audio fuison(new)
    concat_side_by_side("/home/student/hwu/Workplace/AiOS/-9Kic0QsCYs_4-8-rgb_front/-9Kic0QsCYs_4-8-rgb_front.mp4","/data/hwu/how2sign/raw_videos_train/-9Kic0QsCYs_4-8-rgb_front.mp4") #text+audio fuison(new)


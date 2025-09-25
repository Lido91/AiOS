from moviepy.editor import VideoFileClip, clips_array

def concat_side_by_side(video1_path, video2_path, output_path="output_side_by_side.mp4"):
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
    concat_side_by_side("/home/student/hwu/Workplace/ASLDataset_Annotation/dataset/clip/Zpitq3l9il4-022.mp4","/home/student/hwu/Workplace/AiOS/demo/demo_asl/Zpitq3l9il4-022/Zpitq3l9il4-022_demo.mp4","merged_mp4s/Zpitq3l9il4-022.mp4")
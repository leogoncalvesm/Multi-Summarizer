from summarizer.components.summarizer import HSMVideoSumm
from summarizer.processing.utils import log, process_arguments
from summarizer.processing.dataset import Dataset, DatasetLoader


def main(**kwargs):
    output = kwargs.pop("output")
    dataset = Dataset(**kwargs)
    log(
        f"""Running for:
    - Dataset: {dataset.name}
    - Path: {dataset.path}
    - Videos: {dataset.videos}\n"""
    )

    # Loading data to summarize
    dataset_loader = DatasetLoader(dataset)

    # Saving frames as images
    frames_dir = dataset_loader.save_video_frames()

    # Loading videos
    videos = dataset_loader.load_videos()

    # Creating summarizer object
    video_summ = HSMVideoSumm(
        summary_name=dataset.name,
        frames_path=frames_dir,
        output_path=output,
        videos=videos,
    )

    # Running summarization steps
    video_summ = video_summ.introduction().subjectivity(include=False).redundancy()
    video_summ.save_summary()


if __name__ == "__main__":
    try:
        run_args = process_arguments()
    except Exception as e:
        log(str(e), log_type="ERROR")
    else:
        main(**run_args)

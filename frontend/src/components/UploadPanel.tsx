import {
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  DragEvent,
  KeyboardEvent,
  MouseEvent,
} from "react";

import {
  CloudUpload,
} from "lucide-react";

type Props = {
  onUpload: (files: File[]) => void;
};

export default function UploadPanel({
  onUpload,
}: Props) {
  const inputRef =
    useRef<HTMLInputElement | null>(null);

  const dragCounter =
    useRef(0);

  const [
    isDragging,
    setIsDragging,
  ] = useState(false);

  function openFilePicker() {
    inputRef.current?.click();
  }

  const SUPPORTED_EXTENSIONS = [
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
  ];

   function processFiles(files: File[]) {
      const valid = files.filter((file) => {
      const name = file.name.toLowerCase();

      return SUPPORTED_EXTENSIONS.some(
        (extension) =>
          name.endsWith(extension)
      );
    });

    if (valid.length !== files.length) {
      alert(
        "Поддерживаются MP4, MOV, AVI, MKV, TS, MTS, M2TS, MPG и MPEG."
      );
    }

    if (valid.length > 0) {
      onUpload(valid);
    }
  }

  function handleChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    processFiles(
      Array.from(
        event.target.files ?? []
      )
    );

    event.target.value = "";
  }

  function handleDragEnter(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    dragCounter.current += 1;

    setIsDragging(true);
  }

  function handleDragOver(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    event.dataTransfer.dropEffect =
      "copy";
  }

  function handleDragLeave(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    dragCounter.current -= 1;

    if (
      dragCounter.current <= 0
    ) {
      dragCounter.current = 0;
      setIsDragging(false);
    }
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    dragCounter.current = 0;
    setIsDragging(false);

    const files =
      Array.from(
        event.dataTransfer.files
      );

    processFiles(files);
  }

  function handleKeyDown(
    event: KeyboardEvent<HTMLDivElement>
  ) {
    if (
      event.key === "Enter" ||
      event.key === " "
    ) {
      event.preventDefault();

      openFilePicker();
    }
  }

  function handleButtonClick(
    event: MouseEvent<HTMLButtonElement>
  ) {
    event.stopPropagation();

    openFilePicker();
  }

  return (
    <div
      className={`upload-card ${
        isDragging
          ? "upload-card-active"
          : ""
      }`}
      role="button"
      tabIndex={0}
      onClick={
        openFilePicker
      }
      onKeyDown={
        handleKeyDown
      }
      onDragEnter={
        handleDragEnter
      }
      onDragOver={
        handleDragOver
      }
      onDragLeave={
        handleDragLeave
      }
      onDrop={
        handleDrop
      }
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        multiple
  accept=".mov,.mp4,.avi,.mkv,.ts,.mts,.m2ts,.mpg,.mpeg,video/*"
        onChange={
          handleChange
        }
      />

      <CloudUpload
        className="upload-main-icon"
        size={48}
        strokeWidth={1.7}
      />

      <button
        className="primary-button"
        type="button"
        onClick={
          handleButtonClick
        }
      >
        <CloudUpload
          size={21}
          strokeWidth={2}
        />

        Выбрать файлы
      </button>

  <div className="format-row">
    <span>MP4</span>
    <span>MOV</span>
    <span>AVI</span>
    <span>MKV</span>
    <span>TS</span>
  </div>
    </div>
  );
}

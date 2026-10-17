import {
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  DragEvent,
} from "react";

type Props = {
  onUpload: (files: File[]) => void;
  disabled?: boolean;
};

export default function UploadPanel({
  onUpload,
  disabled = false,
}: Props) {
  const inputRef =
    useRef<HTMLInputElement | null>(null);

  const [dragging, setDragging] =
    useState(false);

  function validateFiles(
    files: File[]
  ): File[] {
    const validFiles =
      files.filter((file) => {
        const name =
          file.name.toLowerCase();

        return (
          name.endsWith(".mov") ||
          name.endsWith(".mp4")
        );
      });

    if (
      validFiles.length !==
      files.length
    ) {
      alert(
        "Некоторые файлы были пропущены. Разрешены только .mov и .mp4."
      );
    }

    return validFiles;
  }

  function handleFiles(
    files: File[]
  ) {
    const valid =
      validateFiles(files);

    if (valid.length === 0) {
      return;
    }

    onUpload(valid);
  }

  function handleChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const files = Array.from(
      event.target.files ?? []
    );

    handleFiles(files);

    // Позволяет повторно выбрать
    // тот же файл
    event.target.value = "";
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    setDragging(false);

    if (disabled) {
      return;
    }

    const files = Array.from(
      event.dataTransfer.files
    );

    handleFiles(files);
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Загрузить одно или несколько видео"
      aria-disabled={disabled}
      onKeyDown={(event) => {
        if (!disabled && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      className={`upload ${
        dragging
          ? "upload-dragging"
          : ""
      }`}
      onClick={() => {
        if (!disabled) {
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();

        if (!disabled) {
          setDragging(true);
        }
      }}
      onDragLeave={() =>
        setDragging(false)
      }
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        disabled={disabled}
        multiple
        accept=".mov,.mp4,video/quicktime,video/mp4"
        onChange={handleChange}
      />

      <div className="upload-icon">
        ↑
      </div>

      <h2>Загрузить видео</h2>

      <p>
        Выберите одно или несколько
        видео для параллельного анализа
      </p>

      <small>
        MOV / MP4
      </small>
    </div>
  );
}

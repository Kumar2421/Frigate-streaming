import { useEffect, useMemo, useState } from "react";
import { isDesktop, isIOS, isMobileOnly, isSafari } from "react-device-detect";
import useSWR from "swr";
import { useApiHost } from "@/api";
import { cn } from "@/lib/utils";
import { BsArrowRightCircle } from "react-icons/bs";
import { useNavigate } from "react-router-dom";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TooltipPortal } from "@radix-ui/react-tooltip";
import { SearchResult } from "@/types/search";
import ImageLoadingIndicator from "@/components/indicators/ImageLoadingIndicator";
import useImageLoaded from "@/hooks/use-image-loaded";
import ActivityIndicator from "@/components/indicators/activity-indicator";
import { useTrackedObjectUpdate } from "@/api/ws";
import { isEqual } from "lodash";
import TimeAgo from "@/components/dynamic/TimeAgo";
import SearchResultActions from "@/components/menu/SearchResultActions";
import { SearchTab } from "@/components/overlay/detail/SearchDetailDialog";
import { FrigateConfig } from "@/types/frigateConfig";
import { useTranslation } from "react-i18next";
import { getTranslatedLabel } from "@/utils/i18n";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type CropKind = "first-person" | "first-face";
type CropWhich = "first" | "best" | "both";

type CropItem = {
  kind: CropKind;
  camera: string;
  file: string;
  mtime: number;
  url: string;
  label?: string | null;
  reid_id?: string | number | null;
  score?: number | null;
  which?: "first" | "best" | null;
  frame_time?: number | null;
  timestamp_ms?: number | null;
};

type CropsResponse = {
  kind: CropKind;
  camera?: string | null;
  which: CropWhich;
  items: CropItem[];
};

type ExploreViewProps = {
  searchDetail: SearchResult | undefined;
  setSearchDetail: (search: SearchResult | undefined) => void;
  setSimilaritySearch: (search: SearchResult) => void;
  onSelectSearch: (item: SearchResult, ctrl: boolean, page?: SearchTab) => void;
};

export default function ExploreView({
  searchDetail,
  setSearchDetail,
  setSimilaritySearch,
  onSelectSearch,
}: ExploreViewProps) {
  const { t } = useTranslation(["views/explore"]);
  const apiHost = useApiHost();
  const [selectedCrop, setSelectedCrop] = useState<CropItem | null>(null);
  const [isCropDialogOpen, setIsCropDialogOpen] = useState(false);
  // title

  useEffect(() => {
    document.title = t("documentTitle");
  }, [t]);

  // data

  const {
    data: events,
    mutate,
    isLoading,
    isValidating,
  } = useSWR<SearchResult[]>(
    [
      "events/explore",
      {
        limit: isMobileOnly ? 5 : 10,
      },
    ],
    {
      revalidateOnFocus: true,
    },
  );

  const { data: firstPersonCrops } = useSWR<CropsResponse>(
    [
      "crops/first-person",
      {
        which: "both" as CropWhich,
        limit: isMobileOnly ? 5 : 10,
      },
    ],
    {
      refreshInterval: 5000,
      revalidateOnFocus: true,
    },
  );

  const { data: firstFaceCrops } = useSWR<CropsResponse>(
    [
      "crops/first-face",
      {
        which: "both" as CropWhich,
        limit: isMobileOnly ? 5 : 10,
      },
    ],
    {
      refreshInterval: 5000,
      revalidateOnFocus: true,
    },
  );

  const eventsByLabel = useMemo(() => {
    if (!events) return {};
    return events.reduce<Record<string, SearchResult[]>>((acc, event) => {
      const label = event.label || "Unknown";
      if (!acc[label]) {
        acc[label] = [];
      }
      acc[label].push(event);
      return acc;
    }, {});
  }, [events]);

  const { payload: wsUpdate } = useTrackedObjectUpdate();

  useEffect(() => {
    if (wsUpdate && wsUpdate.type == "description") {
      mutate();
    }
  }, [wsUpdate, mutate]);

  // update search detail when results change

  useEffect(() => {
    if (searchDetail && events) {
      const updatedSearchDetail = events.find(
        (result) => result.id === searchDetail.id,
      );

      if (updatedSearchDetail && !isEqual(updatedSearchDetail, searchDetail)) {
        setSearchDetail(updatedSearchDetail);
      }
    }
  }, [events, searchDetail, setSearchDetail]);

  if (isLoading) {
    return (
      <ActivityIndicator className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2" />
    );
  }

  return (
    <div className="mx-2 space-y-4">
      <Dialog open={isCropDialogOpen} onOpenChange={setIsCropDialogOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>
              {selectedCrop
                ? `${selectedCrop.kind} • ${selectedCrop.camera}`
                : ""}
            </DialogTitle>
          </DialogHeader>
          {selectedCrop && (
            <div className="flex w-full flex-col gap-4 md:flex-row">
              <div className="w-full space-y-3 md:w-1/3">
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Label</div>
                  <div className="text-base font-medium smart-capitalize">
                    {selectedCrop.label || selectedCrop.kind}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Score</div>
                  <div className="text-base font-medium">
                    {typeof selectedCrop.score === "number"
                      ? `${Math.round(selectedCrop.score * 100)}%`
                      : "-"}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">ReID</div>
                  <div className="text-base font-medium">
                    {selectedCrop.reid_id ?? "-"}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Camera</div>
                  <div className="text-base font-medium smart-capitalize">
                    {selectedCrop.camera}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">When</div>
                  <div className="text-base font-medium">
                    <TimeAgo
                      time={
                        (selectedCrop.timestamp_ms ??
                          (typeof selectedCrop.frame_time === "number"
                            ? selectedCrop.frame_time * 1000
                            : selectedCrop.mtime * 1000))
                      }
                      dense
                    />
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(
                      (selectedCrop.timestamp_ms ??
                        (typeof selectedCrop.frame_time === "number"
                          ? selectedCrop.frame_time * 1000
                          : selectedCrop.mtime * 1000))
                    ).toLocaleString()}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Which</div>
                  <div className="text-base font-medium">
                    {selectedCrop.which || "-"}
                  </div>
                </div>
              </div>

              <div className="flex w-full items-center justify-center md:w-2/3">
                <img
                  className="max-h-[70vh] w-auto rounded-lg object-contain"
                  src={`${apiHost}${selectedCrop.url}?t=${selectedCrop.mtime}`}
                  alt={`${selectedCrop.kind} crop`}
                />
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
      <CropsThumbnailRow
        title="First Person"
        kind="first-person"
        crops={firstPersonCrops?.items}
        onSelectCrop={(crop) => {
          setSelectedCrop(crop);
          setIsCropDialogOpen(true);
        }}
      />
      <CropsThumbnailRow
        title="First Face"
        kind="first-face"
        crops={firstFaceCrops?.items}
        onSelectCrop={(crop) => {
          setSelectedCrop(crop);
          setIsCropDialogOpen(true);
        }}
      />
      {Object.entries(eventsByLabel).map(([label, filteredEvents]) => (
        <ThumbnailRow
          key={label}
          searchResults={filteredEvents}
          isValidating={isValidating}
          objectType={label}
          setSearchDetail={setSearchDetail}
          mutate={mutate}
          setSimilaritySearch={setSimilaritySearch}
          onSelectSearch={onSelectSearch}
        />
      ))}
    </div>
  );
}

type CropsThumbnailRowProps = {
  title: string;
  kind: CropKind;
  crops?: CropItem[];
  onSelectCrop: (crop: CropItem) => void;
};

function CropsThumbnailRow({ title, kind, crops, onSelectCrop }: CropsThumbnailRowProps) {
  const apiHost = useApiHost();
  const navigate = useNavigate();
  const { t } = useTranslation(["views/explore"]);

  if (!crops || crops.length === 0) {
    return null;
  }

  return (
    <div className="rounded-lg bg-background_alt p-2 md:px-4">
      <div className="flex flex-row items-center text-lg smart-capitalize">
        {title}
      </div>
      <div className="flex flex-row items-center space-x-2 py-2">
        {crops.map((c) => (
          <div
            key={`${c.camera}-${c.file}`}
            className="relative aspect-square h-auto max-w-[20%] flex-grow md:max-w-[10%]"
          >
            <CropThumbnailImage
              apiHost={apiHost}
              crop={c}
              onSelect={() => onSelectCrop(c)}
            />
          </div>
        ))}
        <div
          className="flex cursor-pointer items-center justify-center"
          onClick={() => {
            const params = new URLSearchParams({ crops: kind }).toString();
            navigate(`/explore?${params}`);
          }}
        >
          <Tooltip>
            <TooltipTrigger>
              <BsArrowRightCircle
                className="ml-2 text-secondary-foreground transition-all duration-300 hover:text-primary"
                size={24}
              />
            </TooltipTrigger>
            <TooltipPortal>
              <TooltipContent>
                {t("exploreMore", { label: title })}
              </TooltipContent>
            </TooltipPortal>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}

type CropThumbnailImageProps = {
  apiHost: string;
  crop: CropItem;
  onSelect: () => void;
};

function CropThumbnailImage({ apiHost, crop, onSelect }: CropThumbnailImageProps) {
  const [imgRef, imgLoaded, onImgLoad] = useImageLoaded();
  const src = `${apiHost}${crop.url}?t=${crop.mtime}`;

  return (
    <div
      className="relative size-full"
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          onSelect();
        }
      }}
    >
      <ImageLoadingIndicator
        className="absolute inset-0"
        imgLoaded={imgLoaded}
      />
      <img
        ref={imgRef}
        className={cn(
          "absolute size-full cursor-pointer rounded-lg object-cover transition-all duration-300 ease-in-out lg:rounded-2xl",
          !imgLoaded && "invisible",
        )}
        style={
          isIOS
            ? {
              WebkitUserSelect: "none",
              WebkitTouchCallout: "none",
            }
            : undefined
        }
        loading={isSafari ? "eager" : "lazy"}
        draggable={false}
        src={src}
        onLoad={onImgLoad}
        alt={`${crop.kind} crop`}
      />
      {isDesktop && (
        <div className="absolute bottom-1 right-1 z-10 rounded-lg bg-black/50 px-2 py-1 text-xs text-white">
          {crop.camera}
        </div>
      )}
    </div>
  );
}

type ThumbnailRowType = {
  objectType: string;
  searchResults?: SearchResult[];
  isValidating: boolean;
  setSearchDetail: (search: SearchResult | undefined) => void;
  mutate: () => void;
  setSimilaritySearch: (search: SearchResult) => void;
  onSelectSearch: (item: SearchResult, ctrl: boolean, page?: SearchTab) => void;
};

function ThumbnailRow({
  objectType,
  searchResults,
  isValidating,
  setSearchDetail,
  mutate,
  setSimilaritySearch,
  onSelectSearch,
}: ThumbnailRowType) {
  const { t } = useTranslation(["views/explore"]);
  const navigate = useNavigate();

  const handleSearch = (label: string) => {
    const similaritySearchParams = new URLSearchParams({
      labels: label,
    }).toString();
    navigate(`/explore?${similaritySearchParams}`);
  };

  return (
    <div className="rounded-lg bg-background_alt p-2 md:px-4">
      <div className="flex flex-row items-center text-lg smart-capitalize">
        {getTranslatedLabel(objectType)}
        {searchResults && (
          <span className="ml-3 text-sm text-secondary-foreground">
            {t("trackedObjectsCount", {
              // @ts-expect-error we know this is correct
              count: searchResults[0].event_count,
            })}
          </span>
        )}
        {isValidating && <ActivityIndicator className="ml-2 size-4" />}
      </div>
      <div className="flex flex-row items-center space-x-2 py-2">
        {searchResults?.map((event) => (
          <div
            key={event.id}
            className="relative aspect-square h-auto max-w-[20%] flex-grow md:max-w-[10%]"
          >
            <ExploreThumbnailImage
              event={event}
              setSearchDetail={setSearchDetail}
              mutate={mutate}
              setSimilaritySearch={setSimilaritySearch}
              onSelectSearch={onSelectSearch}
            />
          </div>
        ))}
        <div
          className="flex cursor-pointer items-center justify-center"
          onClick={() => handleSearch(objectType)}
        >
          <Tooltip>
            <TooltipTrigger>
              <BsArrowRightCircle
                className="ml-2 text-secondary-foreground transition-all duration-300 hover:text-primary"
                size={24}
              />
            </TooltipTrigger>
            <TooltipPortal>
              <TooltipContent>
                {t("exploreMore", { label: getTranslatedLabel(objectType) })}
              </TooltipContent>
            </TooltipPortal>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}

type ExploreThumbnailImageProps = {
  event: SearchResult;
  setSearchDetail: (search: SearchResult | undefined) => void;
  mutate: () => void;
  setSimilaritySearch: (search: SearchResult) => void;
  onSelectSearch: (item: SearchResult, ctrl: boolean, page?: SearchTab) => void;
};
function ExploreThumbnailImage({
  event,
  setSearchDetail,
  mutate,
  setSimilaritySearch,
  onSelectSearch,
}: ExploreThumbnailImageProps) {
  const apiHost = useApiHost();
  const { data: config } = useSWR<FrigateConfig>("config");
  const [imgRef, imgLoaded, onImgLoad] = useImageLoaded();
  const navigate = useNavigate();

  const handleFindSimilar = () => {
    if (config?.semantic_search.enabled) {
      setSimilaritySearch(event);
    }
  };

  const handleShowObjectLifecycle = () => {
    onSelectSearch(event, false, "object_lifecycle");
  };

  const handleShowSnapshot = () => {
    onSelectSearch(event, false, "snapshot");
  };

  const handleAddTrigger = () => {
    navigate(
      `/settings?page=triggers&camera=${event.camera}&event_id=${event.id}`,
    );
  };

  return (
    <SearchResultActions
      searchResult={event}
      findSimilar={handleFindSimilar}
      refreshResults={mutate}
      showObjectLifecycle={handleShowObjectLifecycle}
      showSnapshot={handleShowSnapshot}
      addTrigger={handleAddTrigger}
      isContextMenu={true}
    >
      <div className="relative size-full">
        <ImageLoadingIndicator
          className="absolute inset-0"
          imgLoaded={imgLoaded}
        />
        <img
          ref={imgRef}
          className={cn(
            "absolute size-full cursor-pointer rounded-lg object-cover transition-all duration-300 ease-in-out lg:rounded-2xl",
            !imgLoaded && "invisible",
          )}
          style={
            isIOS
              ? {
                WebkitUserSelect: "none",
                WebkitTouchCallout: "none",
              }
              : undefined
          }
          loading={isSafari ? "eager" : "lazy"}
          draggable={false}
          src={`${apiHost}api/events/${event.id}/thumbnail.webp`}
          onClick={() => setSearchDetail(event)}
          onLoad={onImgLoad}
          alt={`${event.label} thumbnail`}
        />
        {isDesktop && (
          <div className="absolute bottom-1 right-1 z-10 rounded-lg bg-black/50 px-2 py-1 text-xs text-white">
            {event.end_time ? (
              <TimeAgo time={event.start_time * 1000} dense />
            ) : (
              <div>
                <ActivityIndicator size={10} />
              </div>
            )}
          </div>
        )}
      </div>
    </SearchResultActions>
  );
}

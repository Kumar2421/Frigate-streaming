import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
    Users,
    UserCheck,
    Clock,
    TrendingUp,
    Eye,
    EyeOff
} from "lucide-react";

interface ReIDMatch {
    matched_track_id: string;
    similarity: number;
    timestamp: number;
}

interface ReIDData {
    [trackId: string]: ReIDMatch[];
}

interface ReIdentificationPanelProps {
    reidData?: ReIDData;
    className?: string;
}

export default function ReIdentificationPanel({
    reidData = {},
    className = "",
}: ReIdentificationPanelProps) {
    const { t } = useTranslation();
    const [expandedTracks, setExpandedTracks] = useState<Set<string>>(new Set());
    const [showAllMatches, setShowAllMatches] = useState(false);

    const formatTimestamp = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleTimeString();
    };

    const getSimilarityColor = (similarity: number) => {
        if (similarity >= 0.9) return "bg-green-500";
        if (similarity >= 0.8) return "bg-yellow-500";
        if (similarity >= 0.7) return "bg-orange-500";
        return "bg-red-500";
    };

    const getSimilarityText = (similarity: number) => {
        if (similarity >= 0.9) return "Very High";
        if (similarity >= 0.8) return "High";
        if (similarity >= 0.7) return "Medium";
        return "Low";
    };

    const toggleTrackExpansion = (trackId: string) => {
        const newExpanded = new Set(expandedTracks);
        if (newExpanded.has(trackId)) {
            newExpanded.delete(trackId);
        } else {
            newExpanded.add(trackId);
        }
        setExpandedTracks(newExpanded);
    };

    const totalMatches = Object.values(reidData).reduce(
        (sum, matches) => sum + matches.length,
        0
    );

    const uniqueTracks = Object.keys(reidData).length;

    const sortedTracks = Object.entries(reidData).sort(
        ([, a], [, b]) => b.length - a.length
    );

    const displayedTracks = showAllMatches
        ? sortedTracks
        : sortedTracks.slice(0, 5);

    if (Object.keys(reidData).length === 0) {
        return (
            <Card className={className}>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Users className="h-5 w-5" />
                        {t("reid.title")}
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="text-center text-muted-foreground py-8">
                        <UserCheck className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>{t("reid.noMatches")}</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className={className}>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Users className="h-5 w-5" />
                        {t("reid.title")}
                    </div>
                    <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" />
                            {totalMatches} {t("reid.matches")}
                        </Badge>
                        <Badge variant="outline">
                            {uniqueTracks} {t("reid.tracks")}
                        </Badge>
                    </div>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-96">
                    <div className="space-y-4">
                        {displayedTracks.map(([trackId, matches]) => (
                            <div key={trackId} className="border rounded-lg p-4">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <Badge variant="outline" className="font-mono text-xs">
                                            {trackId}
                                        </Badge>
                                        <Badge variant="secondary">
                                            {matches.length} {t("reid.matches")}
                                        </Badge>
                                    </div>
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => toggleTrackExpansion(trackId)}
                                        className="h-8 w-8 p-0"
                                    >
                                        {expandedTracks.has(trackId) ? (
                                            <EyeOff className="h-4 w-4" />
                                        ) : (
                                            <Eye className="h-4 w-4" />
                                        )}
                                    </Button>
                                </div>

                                {expandedTracks.has(trackId) && (
                                    <div className="space-y-2">
                                        <Separator />
                                        {matches.map((match, index) => (
                                            <div
                                                key={index}
                                                className="flex items-center justify-between p-3 bg-muted/50 rounded-md"
                                            >
                                                <div className="flex items-center gap-3">
                                                    <div
                                                        className={`w-3 h-3 rounded-full ${getSimilarityColor(
                                                            match.similarity
                                                        )}`}
                                                    />
                                                    <div>
                                                        <p className="font-mono text-sm">
                                                            {match.matched_track_id}
                                                        </p>
                                                        <p className="text-xs text-muted-foreground">
                                                            {getSimilarityText(match.similarity)} confidence
                                                        </p>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <p className="text-sm font-medium">
                                                        {(match.similarity * 100).toFixed(1)}%
                                                    </p>
                                                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                                                        <Clock className="h-3 w-3" />
                                                        {formatTimestamp(match.timestamp)}
                                                    </p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {sortedTracks.length > 5 && (
                        <div className="mt-4 pt-4 border-t">
                            <Button
                                variant="outline"
                                onClick={() => setShowAllMatches(!showAllMatches)}
                                className="w-full"
                            >
                                {showAllMatches
                                    ? t("reid.showLess")
                                    : t("reid.showMore", { count: sortedTracks.length - 5 })}
                            </Button>
                        </div>
                    )}
                </ScrollArea>
            </CardContent>
        </Card>
    );
}

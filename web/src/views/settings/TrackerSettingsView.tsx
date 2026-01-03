import { useState } from "react";
import { useTranslation } from "react-i18next";
import useSWR from "swr";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
    Brain,
    Target,
    Zap,
    Info,
    AlertTriangle
} from "lucide-react";
import { FrigateConfig } from "@/types/frigateConfig";

export default function TrackerSettingsView() {
    const { t } = useTranslation();
    const { data: config, mutate: updateConfig } = useSWR<FrigateConfig>("config");
    const [isLoading, setIsLoading] = useState(false);

    const handleSave = async () => {
        if (!config) return;

        setIsLoading(true);
        try {
            const response = await fetch("api/config", {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    ...config,
                    tracker: config.tracker,
                }),
            });

            if (response.ok) {
                await updateConfig();
                // Show success message
            } else {
                // Show error message
            }
        } catch (error) {
            console.error("Failed to save tracker configuration:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const updateTrackerType = (type: "norfair" | "centroid" | "deepocsort") => {
        if (!config) return;

        const newConfig = {
            ...config,
            tracker: {
                ...config.tracker,
                type,
            },
        };

        updateConfig(newConfig, false);
    };

    const updateDeepOCSORTConfig = (field: string, value: any) => {
        if (!config) return;

        const newConfig = {
            ...config,
            tracker: {
                ...config.tracker,
                deepocsort: {
                    ...config.tracker.deepocsort,
                    [field]: value,
                },
            },
        } as FrigateConfig;

        updateConfig(newConfig, false);
    };

    if (!config) {
        return <div>Loading...</div>;
    }

    const isDeepOCSORT = config.tracker?.type === "deepocsort";

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold tracking-tight">{t("tracker.title")}</h1>
                    <p className="text-muted-foreground">
                        {t("tracker.description")}
                    </p>
                </div>
                <Button onClick={handleSave} disabled={isLoading}>
                    {isLoading ? t("common.saving") : t("common.save")}
                </Button>
            </div>

            <div className="grid gap-6">
                {/* Tracker Type Selection */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Target className="h-5 w-5" />
                            {t("tracker.type.title")}
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="tracker-type">{t("tracker.type.label")}</Label>
                            <Select
                                value={config.tracker?.type || "norfair"}
                                onValueChange={updateTrackerType}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder={t("tracker.type.placeholder")} />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="norfair">
                                        <div className="flex items-center gap-2">
                                            <Badge variant="outline">Norfair</Badge>
                                            <span>{t("tracker.type.norfair")}</span>
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="centroid">
                                        <div className="flex items-center gap-2">
                                            <Badge variant="outline">Centroid</Badge>
                                            <span>{t("tracker.type.centroid")}</span>
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="deepocsort">
                                        <div className="flex items-center gap-2">
                                            <Badge variant="default">DeepOCSORT</Badge>
                                            <span>{t("tracker.type.deepocsort")}</span>
                                        </div>
                                    </SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {isDeepOCSORT && (
                            <Alert>
                                <Info className="h-4 w-4" />
                                <AlertDescription>
                                    {t("tracker.deepocsort.info")}
                                </AlertDescription>
                            </Alert>
                        )}
                    </CardContent>
                </Card>

                {/* DeepOCSORT Configuration */}
                {isDeepOCSORT && (
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Brain className="h-5 w-5" />
                                {t("tracker.deepocsort.title")}
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            {/* Basic Parameters */}
                            <div className="space-y-4">
                                <h4 className="text-sm font-medium">{t("tracker.deepocsort.basic")}</h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="det-thresh">{t("tracker.deepocsort.det_thresh")}</Label>
                                        <Input
                                            id="det-thresh"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.det_thresh || 0.3}
                                            onChange={(e) => updateDeepOCSORTConfig("det_thresh", parseFloat(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="max-age">{t("tracker.deepocsort.max_age")}</Label>
                                        <Input
                                            id="max-age"
                                            type="number"
                                            min="1"
                                            value={config.tracker.deepocsort?.max_age || 30}
                                            onChange={(e) => updateDeepOCSORTConfig("max_age", parseInt(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="min-hits">{t("tracker.deepocsort.min_hits")}</Label>
                                        <Input
                                            id="min-hits"
                                            type="number"
                                            min="1"
                                            value={config.tracker.deepocsort?.min_hits || 3}
                                            onChange={(e) => updateDeepOCSORTConfig("min_hits", parseInt(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="iou-threshold">{t("tracker.deepocsort.iou_threshold")}</Label>
                                        <Input
                                            id="iou-threshold"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.iou_threshold || 0.3}
                                            onChange={(e) => updateDeepOCSORTConfig("iou_threshold", parseFloat(e.target.value))}
                                        />
                                    </div>
                                </div>
                            </div>

                            <Separator />

                            {/* Re-identification Parameters */}
                            <div className="space-y-4">
                                <h4 className="text-sm font-medium flex items-center gap-2">
                                    <Zap className="h-4 w-4" />
                                    {t("tracker.deepocsort.reid")}
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="reid-model">{t("tracker.deepocsort.reid_model")}</Label>
                                        <Select
                                            value={config.tracker.deepocsort?.reid_model_path || "osnet_x1_0"}
                                            onValueChange={(value) => updateDeepOCSORTConfig("reid_model_path", value)}
                                        >
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="osnet_x1_0">OSNet x1.0 (Recommended)</SelectItem>
                                                <SelectItem value="osnet_x0_75">OSNet x0.75 (Faster)</SelectItem>
                                                <SelectItem value="osnet_x0_5">OSNet x0.5 (Fastest)</SelectItem>
                                                <SelectItem value="resnet50">ResNet50 (Fallback)</SelectItem>
                                                <SelectItem value="resnet101">ResNet101 (High Accuracy)</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="reid-device">{t("tracker.deepocsort.reid_device")}</Label>
                                        <Select
                                            value={config.tracker.deepocsort?.reid_device || "cpu"}
                                            onValueChange={(value) => updateDeepOCSORTConfig("reid_device", value)}
                                        >
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="cpu">CPU</SelectItem>
                                                <SelectItem value="cuda">CUDA</SelectItem>
                                                <SelectItem value="cuda:0">CUDA:0</SelectItem>
                                                <SelectItem value="cuda:1">CUDA:1</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="reid-threshold">{t("tracker.deepocsort.reid_threshold")}</Label>
                                        <Input
                                            id="reid-threshold"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.reid_threshold || 0.7}
                                            onChange={(e) => updateDeepOCSORTConfig("reid_threshold", parseFloat(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="w-association-emb">{t("tracker.deepocsort.w_association_emb")}</Label>
                                        <Input
                                            id="w-association-emb"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.w_association_emb || 0.75}
                                            onChange={(e) => updateDeepOCSORTConfig("w_association_emb", parseFloat(e.target.value))}
                                        />
                                    </div>
                                </div>
                            </div>

                            <Separator />

                            {/* Advanced Parameters */}
                            <div className="space-y-4">
                                <h4 className="text-sm font-medium">{t("tracker.deepocsort.advanced")}</h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="delta-t">{t("tracker.deepocsort.delta_t")}</Label>
                                        <Input
                                            id="delta-t"
                                            type="number"
                                            min="1"
                                            value={config.tracker.deepocsort?.delta_t || 3}
                                            onChange={(e) => updateDeepOCSORTConfig("delta_t", parseInt(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="inertia">{t("tracker.deepocsort.inertia")}</Label>
                                        <Input
                                            id="inertia"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.inertia || 0.2}
                                            onChange={(e) => updateDeepOCSORTConfig("inertia", parseFloat(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="alpha-fixed-emb">{t("tracker.deepocsort.alpha_fixed_emb")}</Label>
                                        <Input
                                            id="alpha-fixed-emb"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.alpha_fixed_emb || 0.95}
                                            onChange={(e) => updateDeepOCSORTConfig("alpha_fixed_emb", parseFloat(e.target.value))}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="aw-param">{t("tracker.deepocsort.aw_param")}</Label>
                                        <Input
                                            id="aw-param"
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={config.tracker.deepocsort?.aw_param || 0.5}
                                            onChange={(e) => updateDeepOCSORTConfig("aw_param", parseFloat(e.target.value))}
                                        />
                                    </div>
                                </div>
                            </div>

                            <Separator />

                            {/* Feature Toggles */}
                            <div className="space-y-4">
                                <h4 className="text-sm font-medium">{t("tracker.deepocsort.features")}</h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="flex items-center justify-between">
                                        <div className="space-y-0.5">
                                            <Label>{t("tracker.deepocsort.embedding_off")}</Label>
                                            <p className="text-xs text-muted-foreground">
                                                {t("tracker.deepocsort.embedding_off_desc")}
                                            </p>
                                        </div>
                                        <Switch
                                            checked={!config.tracker.deepocsort?.embedding_off}
                                            onCheckedChange={(checked) => updateDeepOCSORTConfig("embedding_off", !checked)}
                                        />
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <div className="space-y-0.5">
                                            <Label>{t("tracker.deepocsort.cmc_off")}</Label>
                                            <p className="text-xs text-muted-foreground">
                                                {t("tracker.deepocsort.cmc_off_desc")}
                                            </p>
                                        </div>
                                        <Switch
                                            checked={!config.tracker.deepocsort?.cmc_off}
                                            onCheckedChange={(checked) => updateDeepOCSORTConfig("cmc_off", !checked)}
                                        />
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <div className="space-y-0.5">
                                            <Label>{t("tracker.deepocsort.aw_off")}</Label>
                                            <p className="text-xs text-muted-foreground">
                                                {t("tracker.deepocsort.aw_off_desc")}
                                            </p>
                                        </div>
                                        <Switch
                                            checked={!config.tracker.deepocsort?.aw_off}
                                            onCheckedChange={(checked) => updateDeepOCSORTConfig("aw_off", !checked)}
                                        />
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <div className="space-y-0.5">
                                            <Label>{t("tracker.deepocsort.new_kf_off")}</Label>
                                            <p className="text-xs text-muted-foreground">
                                                {t("tracker.deepocsort.new_kf_off_desc")}
                                            </p>
                                        </div>
                                        <Switch
                                            checked={!config.tracker.deepocsort?.new_kf_off}
                                            onCheckedChange={(checked) => updateDeepOCSORTConfig("new_kf_off", !checked)}
                                        />
                                    </div>
                                </div>
                            </div>

                            <Alert>
                                <AlertTriangle className="h-4 w-4" />
                                <AlertDescription>
                                    {t("tracker.deepocsort.warning")}
                                </AlertDescription>
                            </Alert>
                        </CardContent>
                    </Card>
                )}
            </div>
        </div>
    );
}

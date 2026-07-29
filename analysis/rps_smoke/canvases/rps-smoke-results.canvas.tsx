import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const EPOCHS = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
].map(String);

const REWARD = [
  0.066, 0.11, 0.086, 0.16, 0.11, 0.176, 0.314, 0.246, 0.218, 0.224, 0.238,
  0.3, 0.26, 0.264, 0.328, 0.342, 0.262, 0.29, 0.31, 0.326,
];

const SP_PCT = [
  15.6, 20.8, 26.2, 25.2, 24.6, 27.8, 44.2, 37.0, 33.2, 35.2, 35.2, 42.4,
  35.2, 36.6, 36.4, 36.8, 30.6, 38.8, 34.2, 34.4,
];

const ENT1 = [
  0.805, 0.822, 0.755, 0.761, 0.699, 0.625, 0.55, 0.6, 0.636, 0.634, 0.57,
  0.5, 0.476, 0.487, 0.444, 0.437, 0.439, 0.406, 0.417, 0.376,
];
const ENT2 = [
  0.675, 0.673, 0.686, 0.714, 0.716, 0.737, 0.683, 0.711, 0.729, 0.641,
  0.741, 0.714, 0.732, 0.735, 0.745, 0.713, 0.74, 0.718, 0.734, 0.759,
];

const KL1 = [
  0.005, 0.019, 0.118, 0.165, 0.212, 0.305, 0.54, 0.455, 0.532, 0.417,
  0.474, 0.635, 0.593, 0.586, 0.565, 0.576, 0.513, 0.618, 0.598, 0.688,
];
const KL2 = [
  0, -0.002, 0.001, -0.001, 0.004, 0.001, -0.002, -0.003, 0.002, 0,
  0, 0.002, 0, 0.006, 0.004, 0.004, 0.007, 0.004, 0.013, 0.004,
];

const A1_R = [
  40.2, 37.2, 30.6, 28.2, 22.4, 20.0, 13.2, 13.2, 10.6, 12.6, 12.2, 10.2,
  12.6, 12.6, 14.2, 12.4, 17.6, 11.0, 12.0, 10.8,
];
const A1_P = [
  25.8, 27.4, 21.2, 25.0, 28.0, 29.2, 19.2, 29.6, 35.0, 37.6, 36.8, 30.6,
  33.0, 31.4, 31.2, 31.6, 31.2, 32.8, 36.0, 38.0,
];
const A1_S = [
  34.0, 35.4, 48.2, 46.8, 49.6, 50.8, 67.6, 57.2, 54.4, 49.8, 51.0, 59.2,
  54.4, 56.0, 54.6, 56.0, 51.2, 56.2, 52.0, 51.2,
];

const A2_R = [
  26.2, 26.4, 23.0, 22.2, 26.8, 25.0, 19.4, 25.8, 21.8, 22.6, 16.8, 16.4,
  21.4, 19.0, 20.8, 21.6, 22.4, 22.8, 19.4, 22.8,
];
const A2_P = [
  45.4, 52.2, 49.8, 51.2, 47.0, 46.4, 59.8, 56.0, 55.4, 61.0, 58.0, 62.0,
  55.8, 57.6, 56.2, 55.8, 52.6, 60.4, 55.8, 55.0,
];
const A2_S = [
  28.4, 21.4, 27.2, 26.6, 26.2, 28.6, 20.8, 18.2, 22.8, 16.4, 25.2, 21.6,
  22.8, 23.4, 23.0, 22.6, 25.0, 16.8, 24.8, 22.2,
];

const VL2 = [
  6.9, 7.6, 10.2, 9.1, 7.0, 16.9, 26.8, 19.5, 16.9, 13.8, 16.4, 25.7,
  16.5, 17.2, 31.1, 30.7, 17.2, 18.1, 26.5, 27.9,
];

export default function RpsSmokeResults() {
  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1100 }}>
      <Stack gap={8}>
        <H1>RPS shaping smoke — results</H1>
        <Text tone="secondary">
          Source: checkpoints/rps_shaping_smoke · configs/rps_shaping.json · 20
          epochs · t_max=20 · e_max=5 · n_games=5 · seed 1 · MPS
        </Text>
        <Row gap={8}>
          <Pill tone="success">0 NaNs in losses</Pill>
          <Pill tone="success">0% illegal actions</Pill>
          <Pill tone="info">100 agent1 updates</Pill>
          <Pill tone="info">20 agent2 updates</Pill>
        </Row>
      </Stack>

      <Callout tone="success" title="Hard-masking fix held">
        Full 20-epoch run completed cleanly. No NaN losses, no multinomial
        crash, and no self-illegal outcomes across 10,000 rounds. The earlier
        PPO ratio NaN path is gone.
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat label="Illegal rate" value="0.00%" tone="success" />
        <Stat label="A1 reward early→late" value="0.11 → 0.31" tone="success" />
        <Stat label="Dominant cell (late)" value="SP 35%" />
        <Stat label="A2 KL (last)" value="0.004" tone="warning" />
      </Grid>

      <Divider />

      <H2>Training dynamics</H2>
      <Text tone="secondary">
        Agent 1 = non-shaper (updates every episode). Agent 2 = shaper (one
        update per trial). Mean reward is from the outcome matrix for agent 1;
        zero-sum so agent 2 ≈ −agent 1.
      </Text>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Mean reward per epoch (agent 1)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 1 env reward", data: REWARD, tone: "success" },
                {
                  name: "SP outcome share (%) / 100",
                  data: SP_PCT.map((x) => x / 100),
                  tone: "info",
                },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Y: reward in [−1, 1]; SP share rescaled to [0, 1] for overlay
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Policy entropy (legal-set)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 1 entropy", data: ENT1, tone: "info" },
                { name: "Agent 2 entropy", data: ENT2, tone: "neutral" },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Uniform over 3 actions ≈ 1.099 nats
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>KL to reference</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 1 KL", data: KL1, tone: "warning" },
                { name: "Agent 2 KL", data: KL2, tone: "neutral" },
              ]}
              beginAtZero
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper value loss</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 2 value loss", data: VL2, tone: "danger" },
              ]}
              beginAtZero
              height={220}
            />
            <Text tone="secondary" size="small">
              Rising VF error with vf_coef=0.001 and LR 1.41e−7
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>Action mix</H2>
      <Text tone="secondary">
        Marginal action frequencies from joint outcomes (legal cells only).
      </Text>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Agent 1 (non-shaper) — R / P / S %</CardHeader>
          <CardBody>
            <BarChart
              categories={EPOCHS}
              series={[
                { name: "Rock", data: A1_R, tone: "neutral" },
                { name: "Paper", data: A1_P, tone: "info" },
                { name: "Scissors", data: A1_S, tone: "success" },
              ]}
              stacked
              normalized
              valueSuffix="%"
              height={240}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Agent 2 (shaper) — R / P / S %</CardHeader>
          <CardBody>
            <BarChart
              categories={EPOCHS}
              series={[
                { name: "Rock", data: A2_R, tone: "neutral" },
                { name: "Paper", data: A2_P, tone: "info" },
                { name: "Scissors", data: A2_S, tone: "success" },
              ]}
              stacked
              normalized
              valueSuffix="%"
              height={240}
            />
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>Readout</H2>
      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>What looks healthy</H3>
          <Text>
            Stability is the main win: finite losses throughout, illegal rate
            stuck at zero, checkpoints written at epochs 10 and 20.
          </Text>
          <Text>
            Agent 1 clearly adapts: Rock collapses (~40% → ~11%), Scissors
            becomes the mode (~34% → ~51%), entropy falls, KL climbs to ~0.7.
          </Text>
          <Text>
            Env reward for agent 1 trends up (~0.07 → ~0.33) as SP (Scissors vs
            Paper) becomes the dominant cell (~16% → ~35%).
          </Text>
        </Stack>
        <Stack gap={8}>
          <H3>What to watch</H3>
          <Text>
            Agent 2 barely leaves the reference (KL ≈ 0). With LR 1.41e−7,
            cliprange 0.0001, and one update per trial, this smoke is mostly
            “non-shaper learns against a near-fixed Paper-biased opponent,”
            not a full shaping story.
          </Text>
          <Text>
            Shaper value loss drifts up (7 → 28). Worth checking whether the
            value head is under-weighted for the longer trial horizon.
          </Text>
          <Text>
            Mean scores from PPO track env reward sign/magnitude, but are
            score-scaled — use outcome rewards above for gameplay readouts.
          </Text>
        </Stack>
      </Grid>

      <Card>
        <CardHeader trailing={<Text tone="secondary" size="small">early = ep 1–5 · late = ep 16–20</Text>}>
          Early vs late snapshot
        </CardHeader>
        <CardBody>
          <Table
            headers={[
              "Metric",
              "Early",
              "Late",
              "Direction",
            ]}
            rows={[
              ["A1 reward", "0.106", "0.306", "up"],
              ["A1 Rock %", "31.7", "12.8", "down"],
              ["A1 Paper %", "25.5", "33.9", "up"],
              ["A1 Scissors %", "42.8", "53.3", "up"],
              ["A2 Paper %", "49.1", "55.9", "up (mild)"],
              ["SP cell %", "22.5", "35.0", "up"],
              ["Illegal %", "0.00", "0.00", "flat"],
              ["A1 entropy (epoch mean)", "0.77", "0.41", "down"],
              ["A2 KL", "~0.00", "~0.004", "flat"],
            ]}
          />
        </CardBody>
      </Card>

      <Text tone="secondary" size="small">
        This is a 20-epoch smoke, not a paper-scale shaping eval — treat
        behavioral trends as directional.
      </Text>
    </Stack>
  );
}

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

const EPOCHS = Array.from({ length: 20 }, (_, i) => String(i + 1));

// Active smoke (rps_shaping_active.json)
const REWARD = [
  0.066, 0.118, 0.098, 0.158, 0.188, 0.208, 0.276, 0.166, 0.108, 0.072, 0.15,
  0.196, 0.092, -0.014, 0.034, 0.054, 0.056, 0.01, -0.054, 0.054,
];
const SP_PCT = [
  15.6, 22.6, 29.0, 28.4, 25.6, 27.0, 36.0, 24.2, 23.4, 21.0, 22.8, 28.0, 17.2,
  17.2, 12.6, 14.0, 12.2, 13.0, 9.4, 10.6,
];
const ENT1 = [
  0.805, 0.829, 0.74, 0.671, 0.623, 0.592, 0.615, 0.625, 0.644, 0.62, 0.61,
  0.611, 0.562, 0.572, 0.59, 0.573, 0.552, 0.544, 0.596, 0.576,
];
const ENT2 = [
  0.676, 0.669, 0.715, 0.749, 0.747, 0.784, 0.759, 0.797, 0.81, 0.816, 0.835,
  0.845, 0.872, 0.881, 0.877, 0.877, 0.868, 0.854, 0.874, 0.856,
];
const KL1 = [
  0.005, 0.044, 0.186, 0.28, 0.32, 0.334, 0.477, 0.395, 0.422, 0.361, 0.452,
  0.539, 0.52, 0.471, 0.303, 0.365, 0.271, 0.268, 0.22, 0.177,
];
const KL2 = [
  0, 0.003, -0.003, 0.004, 0.008, -0.001, 0.005, 0.03, 0.039, 0.056, 0.048,
  0.021, 0.091, 0.075, 0.125, 0.143, 0.133, 0.132, 0.189, 0.164,
];
const VL2 = [
  6.9, 7.0, 11.8, 10.9, 11.6, 16.9, 24.3, 10.2, 6.3, 6.0, 11.1, 12.0, 5.5, 6.6,
  4.4, 4.8, 6.4, 6.5, 6.7, 8.0,
];
const TL2 = [
  0.347, 0.347, 0.592, 0.546, 0.58, 0.845, 1.215, 0.51, 0.316, 0.3, 0.547,
  0.599, 0.284, 0.325, 0.224, 0.239, 0.324, 0.328, 0.348, 0.407,
];
const PL2 = [
  0.0022, -0.0008, 0.0008, -0.0002, 0.0029, 0.0017, -0.0025, -0.0011, 0.0008,
  -0.0012, -0.0072, -0.0031, 0.007, -0.0035, 0.0014, -0.0002, 0.0035, 0.0037,
  0.0108, 0.0049,
];
const MS2 = [
  -0.079, -0.144, -0.119, -0.191, -0.23, -0.256, -0.34, -0.204, -0.133, -0.089,
  -0.185, -0.243, -0.114, 0.017, -0.042, -0.067, -0.07, -0.012, 0.067, -0.067,
];
// Zero-sum: shaper env reward = -agent1 env reward
const REWARD_A2 = REWARD.map((r) => -r);
const A2_WIN = [
  31.8, 27.0, 30.8, 27.0, 23.2, 22.8, 21.4, 26.0, 27.4, 28.4, 25.4, 23.4, 25.6,
  32.8, 30.0, 27.6, 29.8, 32.2, 33.6, 30.8,
];
const A2_LOSE = [
  38.4, 38.8, 40.6, 42.8, 42.0, 43.6, 49.0, 42.6, 38.2, 35.6, 40.4, 43.0, 34.8,
  31.4, 33.4, 33.0, 35.4, 33.2, 28.2, 36.2,
];
const A2_TIE = [
  29.8, 34.2, 28.6, 30.2, 34.8, 33.6, 29.6, 31.4, 34.4, 36.0, 34.2, 33.6, 39.6,
  35.8, 36.6, 39.4, 34.8, 34.6, 38.2, 33.0,
];
const A1_R = [
  40.2, 33.4, 25.6, 23.6, 18.6, 17.8, 13.4, 14.8, 13.8, 14.2, 12.4, 8.4, 12.4,
  12.0, 18.6, 19.2, 23.8, 21.0, 23.2, 28.0,
];
const A1_P = [
  25.8, 28.6, 21.4, 20.0, 27.6, 33.0, 26.6, 38.4, 41.8, 43.0, 47.4, 44.2, 44.8,
  48.2, 46.2, 40.8, 41.4, 41.0, 43.2, 42.6,
];
const A1_S = [
  34.0, 38.0, 53.0, 56.4, 53.8, 49.2, 60.0, 46.8, 44.4, 42.8, 40.2, 47.4, 42.8,
  39.8, 35.2, 40.0, 34.8, 38.0, 33.6, 29.4,
];
const A2_R = [
  26.2, 25.2, 23.4, 24.8, 29.6, 26.0, 23.0, 28.8, 25.2, 24.8, 23.8, 23.4, 28.0,
  23.8, 29.2, 27.4, 27.0, 31.8, 32.0, 29.2,
];
const A2_P = [
  45.4, 55.0, 51.2, 47.8, 44.8, 47.6, 54.8, 45.6, 48.4, 47.6, 46.8, 54.2, 40.2,
  41.2, 35.0, 35.2, 34.6, 31.8, 29.8, 31.2,
];
const A2_S = [
  28.4, 19.8, 25.4, 27.4, 25.6, 26.4, 22.2, 25.6, 26.4, 27.6, 29.4, 22.4, 31.8,
  35.0, 35.8, 37.4, 38.4, 36.4, 38.2, 39.6,
];

// Baseline overlays for the key “did the shaper unstick?” plots
const BASE_KL2 = [
  0, -0.002, 0.001, -0.001, 0.004, 0.001, -0.002, -0.003, 0.002, 0, 0, 0.002,
  0, 0.006, 0.004, 0.004, 0.007, 0.004, 0.013, 0.004,
];
const BASE_A2_P = [
  45.4, 52.2, 49.8, 51.2, 47.0, 46.4, 59.8, 56.0, 55.4, 61.0, 58.0, 62.0, 55.8,
  57.6, 56.2, 55.8, 52.6, 60.4, 55.8, 55.0,
];
const BASE_REWARD = [
  0.066, 0.11, 0.086, 0.16, 0.11, 0.176, 0.314, 0.246, 0.218, 0.224, 0.238,
  0.3, 0.26, 0.264, 0.328, 0.342, 0.262, 0.29, 0.31, 0.326,
];
const BASE_VL2 = [
  6.9, 7.6, 10.2, 9.1, 7.0, 16.9, 26.8, 19.5, 16.9, 13.8, 16.4, 25.7, 16.5,
  17.2, 31.1, 30.7, 17.2, 18.1, 26.5, 27.9,
];

export default function RpsSmokeActiveResults() {
  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1100 }}>
      <Stack gap={8}>
        <H1>RPS shaping smoke — active shaper</H1>
        <Text tone="secondary">
          Source: checkpoints/rps_shaping_active_smoke ·
          configs/rps_shaping_active.json · 20 epochs · MPS · compared to
          baseline rps_shaping.json where noted
        </Text>
        <Row gap={8}>
          <Pill tone="success">0 NaNs</Pill>
          <Pill tone="success">0% illegal</Pill>
          <Pill tone="success">A2 KL → 0.16</Pill>
          <Pill tone="info">clip 0.1 · LR 5e-7 · vf 0.05</Pill>
        </Row>
      </Stack>

      <Callout tone="success" title="Shaper actually moved">
        Relaxing clip / LR / vf_coef worked. Agent 2 KL rises to ~0.16 (baseline
        stayed ~0.004), Paper share falls ~49%→33%, and value loss no longer
        explodes. Gameplay is no longer “learner farms a sticky Paper opponent.”
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat label="A2 KL last (base→active)" value="0.004 → 0.16" tone="success" />
        <Stat label="A2 Paper early→late" value="49% → 33%" tone="success" />
        <Stat label="A1 reward early→late" value="0.13 → 0.02" tone="warning" />
        <Stat label="A2 value loss last" value="8.0 vs 27.9" tone="success" />
      </Grid>

      <Divider />

      <H2>Did the hparam change unstick the shaper?</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Agent 2 KL to reference</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Active KL", data: KL2, tone: "success" },
                { name: "Baseline KL", data: BASE_KL2, tone: "neutral" },
              ]}
              beginAtZero
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Agent 2 Paper share (%)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Active Paper %", data: A2_P, tone: "success" },
                { name: "Baseline Paper %", data: BASE_A2_P, tone: "neutral" },
              ]}
              beginAtZero={false}
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Agent 2 value loss</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Active value loss", data: VL2, tone: "info" },
                { name: "Baseline value loss", data: BASE_VL2, tone: "danger" },
              ]}
              beginAtZero
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Agent 1 env reward (zero-sum)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Active A1 reward", data: REWARD, tone: "warning" },
                { name: "Baseline A1 reward", data: BASE_REWARD, tone: "neutral" },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Active late reward drops as the exploitable Paper bias disappears
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>Shaper (agent 2) — deep dive</H2>
      <Text tone="secondary">
        One update per trial. Env reward is −agent-1 reward (zero-sum RPS).
        PPO mean scores are score-scaled batch rewards from the trainer.
      </Text>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Shaper return: env vs PPO mean score</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Env reward (−A1)", data: REWARD_A2, tone: "success" },
                { name: "PPO mean_scores", data: MS2, tone: "info" },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Late env return ~0 — no sustained shaper advantage yet
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper win / lose / tie rate (%)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Win %", data: A2_WIN, tone: "success" },
                { name: "Lose %", data: A2_LOSE, tone: "danger" },
                { name: "Tie %", data: A2_TIE, tone: "neutral" },
              ]}
              beginAtZero={false}
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper action mix over time (%)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Rock %", data: A2_R, tone: "neutral" },
                { name: "Paper %", data: A2_P, tone: "info" },
                { name: "Scissors %", data: A2_S, tone: "success" },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Paper decays; Scissors becomes the mode — not a settled shaping policy
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper policy entropy vs KL</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Entropy (nats)", data: ENT2, tone: "info" },
                { name: "KL to reference", data: KL2, tone: "success" },
              ]}
              beginAtZero
              height={220}
            />
            <Text tone="secondary" size="small">
              Uniform-3 entropy ≈ 1.099; both rise → exploring away from init
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper PPO losses</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Total loss", data: TL2, tone: "neutral" },
                { name: "Value loss × 0.05 (vf_coef scale)", data: VL2.map((v) => v * 0.05), tone: "warning" },
                { name: "Policy loss", data: PL2, tone: "info" },
              ]}
              beginAtZero={false}
              height={220}
            />
            <Text tone="secondary" size="small">
              Value term shown × vf_coef so it is comparable to total loss
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Shaper value loss (raw)</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Active value loss", data: VL2, tone: "info" },
                { name: "Baseline value loss", data: BASE_VL2, tone: "danger" },
              ]}
              beginAtZero
              height={220}
            />
          </CardBody>
        </Card>
      </Grid>

      <Card>
        <CardHeader>Shaper action stack (normalized %)</CardHeader>
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
            height={220}
          />
        </CardBody>
      </Card>

      <Divider />

      <H2>Learner context (agent 1)</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Learner entropy vs shaper entropy</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 1 entropy", data: ENT1, tone: "info" },
                { name: "Agent 2 entropy", data: ENT2, tone: "success" },
              ]}
              beginAtZero={false}
              height={220}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Both KLs + SP exploit cell</CardHeader>
          <CardBody>
            <LineChart
              categories={EPOCHS}
              series={[
                { name: "Agent 1 KL", data: KL1, tone: "warning" },
                { name: "Agent 2 KL", data: KL2, tone: "success" },
                {
                  name: "SP share / 100",
                  data: SP_PCT.map((x) => x / 100),
                  tone: "neutral",
                },
              ]}
              beginAtZero
              height={220}
            />
          </CardBody>
        </Card>
      </Grid>

      <Card>
        <CardHeader>Learner action stack (normalized %)</CardHeader>
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
            height={220}
          />
        </CardBody>
      </Card>

      <Divider />

      <H2>Readout</H2>
      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>What improved</H3>
          <Text>
            Config change did what we wanted: shaper leaves the near-frozen
            regime. KL, action mix, and critic all move; illegal rate stays 0.
          </Text>
          <Text>
            Agent 2 diversifies — Paper down, Scissors up (~25%→40%). That
            removes the easy SP exploit path agent 1 had in the baseline smoke.
          </Text>
        </Stack>
        <Stack gap={8}>
          <H3>What this is not yet</H3>
          <Text>
            Not clear shaping success. Agent 1 late reward collapses toward 0
            as the matchup becomes more mixed — closer to a mutual adaptation /
            cycling regime than “shaper steers learner to a desired equilibrium.”
          </Text>
          <Text>
            Next questions: longer run, whether shaper reward (≈ −A1) trends
            the way you want, and whether trial-level history is being used
            vs plain best-response churn.
          </Text>
        </Stack>
      </Grid>

      <Card>
        <CardHeader trailing="early = ep 1–5 · late = ep 16–20">
          Active vs baseline snapshot
        </CardHeader>
        <CardBody>
          <Table
            headers={["Metric", "Baseline late", "Active late", "Verdict"]}
            rows={[
              ["Illegal %", "0.00", "0.00", "both clean"],
              ["A2 KL", "0.004", "0.164", "unstuck"],
              ["A2 Paper %", "55.9", "32.5", "moved"],
              ["A2 Scissors %", "22.3", "38.0", "moved"],
              ["A1 reward", "0.306", "0.024", "exploit gone"],
              ["SP cell %", "35.0", "11.8", "exploit gone"],
              ["A2 value loss", "27.9", "8.0", "critic healthier"],
              ["A2 entropy", "0.76", "0.86", "more mixed"],
            ]}
          />
        </CardBody>
      </Card>

      <Text tone="secondary" size="small">
        Saved copy also under analysis/rps_smoke/ in the repo, alongside the
        baseline canvas and JSON summaries.
      </Text>
    </Stack>
  );
}

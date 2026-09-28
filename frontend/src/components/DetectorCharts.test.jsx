import { render, screen, fireEvent } from '@testing-library/react';
import { EmotionLineChart } from './DetectorCharts';

describe('EmotionLineChart', () => {
  const mockFrameResults = [
    {
      timestamp: 0,
      result: {
        additional_probs: {
          anger: '0.10',
          happiness: '0.80',
          neutral: '0.05',
          sadness: '0.03',
          surprise: '0.02',
        },
      },
    },
    {
      timestamp: 1,
      result: {
        additional_probs: {
          anger: '0.05',
          happiness: '0.60',
          neutral: '0.25',
          sadness: '0.05',
          surprise: '0.05',
        },
      },
    },
    {
      timestamp: 2,
      result: {
        additional_probs: {
          anger: '0.02',
          happiness: '0.30',
          neutral: '0.50',
          sadness: '0.10',
          surprise: '0.08',
        },
      },
    },
  ];

  it('renders "No data" when frameResults is empty', () => {
    render(<EmotionLineChart frameResults={[]} />);
    expect(screen.getByText('No data')).toBeInTheDocument();
  });

  it('renders "No data" when frameResults is null/undefined', () => {
    render(<EmotionLineChart frameResults={null} />);
    expect(screen.getByText('No data')).toBeInTheDocument();
  });

  it('renders the chart title', () => {
    render(<EmotionLineChart frameResults={mockFrameResults} />);
    expect(screen.getByText('Emotion Timeline')).toBeInTheDocument();
  });

  it('renders the SVG element', () => {
    const { container } = render(<EmotionLineChart frameResults={mockFrameResults} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('renders Y-axis labels (0%, 25%, 50%, 75%, 100%)', () => {
    render(<EmotionLineChart frameResults={mockFrameResults} />);
    expect(screen.getByText('0%')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('renders X-axis time labels', () => {
    render(<EmotionLineChart frameResults={mockFrameResults} />);
    expect(screen.getByText('0.0s')).toBeInTheDocument();
    expect(screen.getByText('2.0s')).toBeInTheDocument();
  });

  it('renders a legend with emotion names', () => {
    render(<EmotionLineChart frameResults={mockFrameResults} />);
    expect(screen.getByText('Anger')).toBeInTheDocument();
    expect(screen.getByText('Happiness')).toBeInTheDocument();
    expect(screen.getByText('Neutral')).toBeInTheDocument();
    expect(screen.getByText('Sadness')).toBeInTheDocument();
    expect(screen.getByText('Surprise')).toBeInTheDocument();
  });

  it('filters out valence and arousal from the chart', () => {
    const withVA = [
      ...mockFrameResults,
      {
        timestamp: 3,
        result: {
          additional_probs: {
            anger: '0.00',
            happiness: '0.00',
            neutral: '1.00',
            sadness: '0.00',
            surprise: '0.00',
            valence: '0.80',
            arousal: '0.30',
          },
        },
      },
    ];
    render(<EmotionLineChart frameResults={withVA} />);
    // Should not render "Valence" or "Arousal" in the chart legend
    expect(screen.queryByText('Valence')).not.toBeInTheDocument();
    expect(screen.queryByText('Arousal')).not.toBeInTheDocument();
  });

  it('renders SVG path elements for emotion lines', () => {
    const { container } = render(<EmotionLineChart frameResults={mockFrameResults} />);
    const paths = container.querySelectorAll('svg path');
    // At least one path per emotion (the line itself), plus area paths
    expect(paths.length).toBeGreaterThanOrEqual(5);
  });

  it('renders nothing when no frames have additional_probs', () => {
    const noProbs = [
      { timestamp: 0, result: {} },
      { timestamp: 1, result: {} },
    ];
    const { container } = render(<EmotionLineChart frameResults={noProbs} />);
    expect(container.innerHTML).toBe('');
  });

  it('handles a single data point without crashing', () => {
    const single = [mockFrameResults[0]];
    const { container } = render(<EmotionLineChart frameResults={single} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
    const paths = svg.querySelectorAll('path');
    // Should still render at least area + line paths
    expect(paths.length).toBeGreaterThanOrEqual(5);
  });

  it('shows tooltip with time on mouse hover', () => {
    const { container } = render(<EmotionLineChart frameResults={mockFrameResults} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();

    // Simulate mouse move over the chart to trigger hover
    if (svg) {
      fireEvent.mouseMove(svg, {
        clientX: 50,
        clientY: 100,
      });
    }

    // After hover, the dashed vertical cursor line should appear
    const dashedLine = container.querySelector('line[stroke-dasharray]');
    expect(dashedLine).toBeInTheDocument();

    // Tooltip SVG rect should appear
    const tooltipRects = container.querySelectorAll('svg rect');
    // More rects than just the tooltip ones... at least check a tooltip rect exists
    const whiteFilledRects = Array.from(container.querySelectorAll('svg rect'))
      .filter(r => r.getAttribute('fill') === 'white');
    expect(whiteFilledRects.length).toBeGreaterThanOrEqual(1);
  });

  it('does not show tooltip initially before hover', () => {
    const { container } = render(<EmotionLineChart frameResults={mockFrameResults} />);
    const dashedLine = container.querySelector('line[stroke-dasharray]');
    expect(dashedLine).not.toBeInTheDocument();
  });

  it('hides tooltip on mouse leave', () => {
    const { container } = render(<EmotionLineChart frameResults={mockFrameResults} />);
    const svg = container.querySelector('svg');

    if (svg) {
      fireEvent.mouseMove(svg, { clientX: 50, clientY: 100 });
    }

    // Should show after hover
    let dashedLine = container.querySelector('line[stroke-dasharray]');
    expect(dashedLine).toBeInTheDocument();

    if (svg) {
      fireEvent.mouseLeave(svg);
    }

    // After mouse leave, the hover line should not be rendered
    dashedLine = container.querySelector('line[stroke-dasharray]');
    expect(dashedLine).not.toBeInTheDocument();
  });
});

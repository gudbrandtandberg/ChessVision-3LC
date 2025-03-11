# Data Collection Guidelines

## Core Principles

1. **Keep Sources Separate**
   - Designate specific days/sources for test data collection
   - Never mix test and training sources
   - Verify distribution consistency across sources

2. **Use Categories, Not Individual Cases**
   - Define challenge categories before collecting data
   - Base categories on objective characteristics
   - Avoid selecting samples based on model performance

## Test Set Guidelines

### DO:
- Define clear challenge categories (e.g., lighting conditions, angles, board states)
- Sample systematically across categories
- Include both challenging and typical cases
- Document selection criteria
- Verify distribution matches real-world scenarios

### DON'T:
- Select samples based on current model failures
- Create artificially difficult test sets
- Mix test and training data sources
- Over-represent edge cases

## Training Set Guidelines

### DO:
- Use validation performance to identify weak categories
- Collect broadly within identified categories
- Maintain class balance
- Include both challenging and typical cases

### DON'T:
- Cherry-pick individual failure cases
- Ignore "easy" cases
- Use test set performance to guide specific sample selection

## Validation Set Guidelines

- Should mirror test set distribution
- Include all categories in natural proportions
- Large enough for reliable metrics
- Separate source from both test and training

## Process for Expanding Datasets

1. **Test Set Expansion**
   1. Analyze current dataset coverage
   2. Identify underrepresented categories
   3. Collect new samples systematically by category
   4. Verify distribution remains representative

2. **Training Set Expansion**
   1. Use validation performance to identify weak categories
   2. Collect new training samples in these categories
   3. Maintain overall dataset balance
   4. Verify separation from test sources

3. **Validation Set Updates**
   1. Mirror test set distribution
   2. Update when test set categories change
   3. Maintain independence from both test and training

## Category Examples

- **Lighting Conditions**
  - Bright
  - Dim
  - Uneven
  - Glare

- **Board Angles**
  - Straight-on
  - Slight tilt
  - Extreme perspective

- **Game States**
  - Opening positions
  - Mid-game complexity
  - Endgame scenarios

- **Environmental Factors**
  - Indoor
  - Outdoor
  - Mixed lighting
  - Background complexity

## Documentation Requirements

1. **For Each Dataset**
   - Source and collection date
   - Category distribution
   - Selection criteria
   - Known biases or limitations

2. **For Each Collection Round**
   - Motivation for collection
   - Categories targeted
   - Selection process
   - Changes in distribution

## Quality Control

1. **Distribution Verification**
   - Check category balance
   - Verify source independence
   - Monitor for unwanted biases

2. **Regular Audits**
   - Review category definitions
   - Verify test/train separation
   - Update documentation

## Special Considerations

- When using day-based splitting:
  - Verify similar distributions across days
  - Include different times of day
  - Consider seasonal factors
  - Document any known variations

Remember: The goal is to create datasets that enable meaningful evaluation and improvement of the model, not to artificially inflate or deflate performance metrics.

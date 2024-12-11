// Common messaging words first
const CHAT_WORDS = [
  'hi', 'hey', 'hello', 'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'you', 
  'bye', 'good', 'great', 'nice', 'cool', 'awesome', 'sorry', 'please', 'how',
  'are', 'is', 'am', 'was', 'were', 'will', 'would', 'can', 'could', 'should',
  'need', 'want', 'like', 'love', 'hate', 'think', 'know', 'see', 'look', 'go',
  'going', 'gone', 'went', 'come', 'coming', 'here', 'there', 'now', 'later',
  'soon', 'maybe', 'probably', 'definitely', 'well', 'fine', 'bad', 'sure'
];

// Common English words
const COMMON_WORDS = [
  'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
  'it', 'for', 'not', 'on', 'with', 'he', 'as', 'do', 'at', 'this',
  'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
  'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
  'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'when', 'make',
  'can', 'time', 'just', 'him', 'take', 'people', 'into', 'year', 'your',
  'some', 'them', 'see', 'other', 'than', 'then', 'now', 'only', 'its',
  'over', 'also', 'use', 'two', 'how', 'our', 'work', 'first', 'well'
];

// Time-related words
const TIME_WORDS = [
  'today', 'tomorrow', 'yesterday', 'now', 'later', 'soon', 'night',
  'morning', 'evening', 'afternoon', 'week', 'month', 'year', 'time',
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
];

export const WORD_LIST = [...new Set([
  ...CHAT_WORDS,
  ...COMMON_WORDS,
  ...TIME_WORDS
])].sort();
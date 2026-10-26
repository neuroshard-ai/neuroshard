"""The user's memory commands, in a chat and from the command line (``neuroshard assistant memory``).

Pure functions over a ``memory.Store`` and an ``ask`` callable, so they are tested without a network or a model.
"""
from neuroshard.assistant import memory

HELP = ('/remember [fact|preference|context:] TEXT   save a note (you confirm it first)\n'
        '/memory                                   list your notes and whether memory is on\n'
        '/forget ID | /forget all                  delete a note, or every note and the key\n'
        '/memory on | /memory off                  turn memory on or off\n'
        '/nomem TEXT                               send this message without any notes\n'
        'Notes go only into the chat message they are chosen for; replies that used them stay in the conversation.')


def listing(store):
    lines = [f"Memory is {'on' if store.enabled else 'off'}. {memory.DISCLOSURE}"]
    items = store.items()
    if not items:
        lines.append('No saved notes.')
    for item in items:
        used = f", last used {item['last_used']}" if item.get('last_used') else ''
        lines.append(f"  {item['id']}  ({item['kind']}) {item['text']}  [saved {item['created']}{used}]")
    return '\n'.join(lines)


def remember(store, text, ask, count_tokens=None):
    """Save one note after showing it and getting a yes. Secrets need an explicit 'yes', not 'y'."""
    kind, _, rest = text.partition(':')
    if kind.strip() in memory.KINDS and rest.strip():
        kind, text = kind.strip(), rest.strip()
    else:
        kind = 'fact'
    shown = memory.clean(text)
    if not shown:
        return 'Nothing to remember. Usage: /remember TEXT'
    size = f', {count_tokens(shown)} tokens' if count_tokens else ''
    lines = [f'Save this {kind}{size}? "{shown}"']
    if not store.enabled:
        lines.append('(Memory is off: the note is saved but not used until you turn it on with /memory on.)')
    warning = memory.secret_check(shown)
    if warning:
        prompt = ('\n'.join(lines) + '\nThis looks like it contains ' + ' and '.join(warning)
                  + '. Saved notes can be sent to network operators. Type "yes" to save it anyway: ')
        if ask(prompt).strip().lower() != 'yes':
            return 'Not saved.'
    elif ask('\n'.join(lines) + ' [y/N] ').strip().lower() not in ('y', 'yes'):
        return 'Not saved.'
    try:
        item = store.add(shown, kind)
    except memory.Invalid as error:
        return f'Not saved: {error}.'
    return f"Saved as {item['id']}."


def forget(store, target, ask):
    target = target.strip()
    if target == 'all':
        if ask('Delete every note and the memory key? This cannot be undone. [y/N] ').strip().lower() not in ('y', 'yes'):
            return 'Nothing deleted.'
        store.wipe()
        return 'All notes and the memory key are deleted.'
    try:
        store.remove(target)
    except memory.Invalid as error:
        return f'{error}.'
    return f'Deleted {target}.'


def command(store, line, ask, count_tokens=None):
    """Handle one memory command typed in a chat. Returns (handled, message, text to send without memory or None)."""
    head, _, rest = line.partition(' ')
    if head == '/remember':
        return True, remember(store, rest, ask, count_tokens), None
    if head == '/forget':
        return True, forget(store, rest, ask), None
    if head == '/memory':
        if rest.strip() in ('on', 'off'):
            store.enable(rest.strip() == 'on')
            return True, f"Memory is {rest.strip()}." + (f' {memory.DISCLOSURE}' if rest.strip() == 'on' else ''), None
        return True, listing(store), None
    if head == '/nomem':
        return (True, None, rest.strip()) if rest.strip() else (True, 'Usage: /nomem TEXT', None)
    if head == '/help':
        return True, HELP, None
    return False, None, None


def disclosure(used):
    """The line printed under a turn that sent notes, or that had to leave relevant notes out."""
    if used and used.get('items'):
        return f"  used memory: {', '.join(used['items'])} ({used['tokens']} tokens sent to this turn's operators)"
    if used and used.get('dropped'):
        return f"  memory not used: {', '.join(used['dropped'])} would not fit this conversation's input limit"
    return None

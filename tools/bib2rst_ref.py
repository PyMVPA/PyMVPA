#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import _bibtex
import re

##REF: Name was automagically refactored
def compare_bib_by_date(a, b):
    """Sorting helper."""
    x = a[1][1]
    y = b[1][1]

    if 'year' in x:
        if 'year' in y:
            if x['year'].isdigit():
                if y['year'].isdigit():
                    # x and y have dates
                    xyear = int( x['year'] )
                    yyear = int( y['year'] )

                    comp =  cmp(xyear, yyear)

                    if comp == 0:
                        return compare_bib_by_author(a,b)
                    else:
                        return (-1)*comp
                else:
                    # x has date, y not -> y is first
                    return 1
            else:
                if y['year'][0].isdigit():
                    return -1
                else:
                    return compare_bib_by_author(a,b)
        else:
            # only x has date
            return 1
    else:
        if 'year' in y:
            return -1
        else:
            # neither nor y have dates
            return compare_bib_by_author(a, b)


##REF: Name was automagically refactored
def compare_bib_by_author(a,b):
    """Sorting helper."""
    x = a[1][1]
    y = b[1][1]

    if 'author' in x:
        if 'author' in y:
            return cmp(join_author_list(x['author']), join_author_list(y['author']))
        else:
            # only x has author
            return 1
    else:
        if 'author' in y:
            return -1
        else:
            # neither nor y have authors
            return 0

##REF: Name was automagically refactored
def format_surname(s, keep_full = False):
    """Recieves a string with surname(s) and returns a string with nicely
    concatenated surnames or initals (with dots).
    """
    # clean spaces
    s = s.strip()

    # go home if empty
    if not len(s):
        return ''

    if not keep_full:
        # only keep initial
        s = s[0]

    if len(s) == 1:
        # add final dot
        s += '.'

    return s


##REF: Name was automagically refactored
def format_author(s, full_surname = False):
    """ Takes a string as argument an tries to determine the lastname and 
    surname(s) of a single author.

    Returns a string with 'lastname, surname(s)'.

    The function takes care of 'von's and other funny prefixes.
    """
    s = s.strip()

    # nothing? take ball, go home
    if not len(s):
        return s

    if s.count(','):
        # assume we have 'lastname, surname(s)'
        slist = s.split(',')
        # take lastname verbatim
        lastname = slist[0].strip()
        # remerge possible surnames with spaces if any
        surnames = u' '.join(slist[1:])

        # get nicely formated surnames concat with spaces
        surname = u' '.join( [ format_surname(i, full_surname) for i in surnames.split() ] )


    else:
        # assume last entity is lastname the rest is surnames
        # check for lastname prefixes
        slist = s.split()
        if len(slist) < 2:
            # only lastname -> finished
            return slist[0]

        # check for order
        if len(slist[-1]) == 1 or slist[-1].endswith('.'):
            # seems like we have lastname->surname order
            if slist[0] in ('von', 'van'):
                lastname = slist[0] + ' ' + slist[1]
                surnames = u' '.join(slist[2:])
            else:
                lastname = slist[0]
                surnames = u' '.join(slist[1:])

        else:
            # the lastname is last
            lastname = slist[-1]

            if slist[-2] in ('von', 'van'):
                lastname = slist[-2] + u' ' + lastname
                surnames = u' '.join(slist[:-2])
            else:
                surnames = u' '.join(slist[:-1])

        surname = u' '.join( [ format_surname(i, full_surname) for i in surnames.split() ] )

    return lastname + u', ' + surname


##REF: Name was automagically refactored
def join_author_list(alist):
    """ Nicely concatenate a list of author with ', ' and a final ' & '.

    Each author is passed to format_author() internally.
    """
    if not len(alist) > 1:
        return format_author(alist[0])

    ret = u', '.join( [ format_author(a) for a in alist[:-1] ] )

    ret += u' & ' + format_author( alist[-1] )

    return ret


##REF: Name was automagically refactored
def format_property(string, indent, max_length = 80):
    """ Helper function to place linebreaks and indentation for
    pretty printing.
    """
    length = len(string)

    lines = []
    pos = 0

    while pos < length:
        if not pos == 0:
            justify = ''.ljust(indent)
            line_length = max_length - indent
        else:
            justify = ''
            line_length = max_length

        if length - pos > line_length:
            lastspace = string.rfind(' ', pos + 1, pos + line_length)
        else:
            lastspace = length

        if lastspace == -1 or lastspace < indent + 1:
            lastspace = string.find(' ', pos + line_length)
            # if no space in the whole string
            if lastspace == -1:
                lastspace = length

        lines.append(justify + string[pos:lastspace])

        pos = lastspace + 1

    return '\n'.join(lines)


class BibTeX(dict):
    """Read bibtex file as dictionary.

    Each entry is accessible by its bibtex ID. An entry is a two-tuple
    `(item_type, dict)`, where `item_type` is eg. article, book, ... and
    `dict` is a dictionary with all bibtex properties for the respective
    item. In this dictionary all properties are store as plain strings,
    except for the list of authors (which is a list of strings) and the pages
    which is a two-tuple with first and last page.
    """
    def __init__(self, filename = None):

        if not filename == None:
            self.open(filename)

        # spaces to be used for indentation
        self.indent = 17

        # maximum line length
        self.line_length = 80


    def open(self, filename):
        """Read and parse bibtex file using python-bibtex."""
        # figure out what the second argument means
        file = _bibtex.open_file(filename, 1)

        while 1:
            entry = _bibtex.next(file)

            if entry == None: break

            eprops = {}

            for k,v in entry[4].iteritems():
                # figure out what the last argument really does
                # leaving in -1 seems to be save
                value = _bibtex.expand(file, v,  0)[2]
                try:
                    value = unicode(value, 'utf-8')
                except UnicodeDecodeError, e:
                    print "ERROR: Failed to decode string '%s'" % value
                    raise
                if k.lower() == 'author':
                    value = value.split(' and ')

                if k.lower() == 'pages':
                    value = tuple(value.replace('-', ' ').split())

                eprops[k] = value

            # bibtex key is dict key
            self[entry[0]] = (entry[1],eprops)


    def __str__(self):
        """Pretty print in bibtex format."""
        bibstring = ''

        for k, v in self.iteritems():
            bibstring += '@' + v[0] + ' { ' + k

            for ek, ev in v[1].iteritems():
                if ek.lower() == 'author':
                    ev = ' and '.join(ev)
                if ek.lower() == 'pages':
                    ev = '--'.join(ev)
                keyname = '  ' + ek

                bibstring += ',\n'
                bibstring += format_property( keyname.ljust(15) + '= {' + ev + '}', 
                                         self.indent,
                                         self.line_length )

            bibstring += "\n}\n\n"


        return bibstring.encode(self.enc)


def bib2rst_references(bib):
    """Compose the reference page."""
    # do it in unicode
    rst = u''
    intro = open('doc/misc/references.in').readlines()
    rst += intro[0]
    rst += "  #\n  # THIS IS A GENERATED FILE -- DO NOT EDIT!\n  #\n"
    rst += ''.join(intro[1:])
    rst += '\n\n'

    biblist = bib.items()
    biblist.sort(compare_bib_by_author)

    for id, (cat, prop) in biblist:
        # put reference target for citations
        rst += '.. _' + id + ':\n\n'

        # compose the citation as the list item label
        cit = u''
        # initial details equal for all item types
        if 'author' in prop:
            cit += u'**' + join_author_list(prop['author']) + u'**'
        if 'year' in prop:
            cit += ' (' + prop['year'] + ').'
        if 'title' in prop:
            cit += ' ' + smooth_rst(prop['title'])
            if not prop['title'].endswith('.'):
                cit += '.'

        # appendix for journal articles
        if cat.lower() == 'article':
            # needs to have journal, volume, pages
            cit += ' *' + prop['journal'] + '*'
            if 'volume' in prop:
                cit += ', *' + prop['volume'] + '*'
            if 'pages' in prop:
                cit += ', ' + '-'.join(prop['pages'])
        elif cat.lower() == 'book':
            # needs to have publisher, address
            cit += ' ' + prop['publisher']
            cit += ': ' + prop['address']
        elif cat.lower() == 'manual':
            cit += ' ' + prop['address']
        elif cat.lower() == 'inproceedings':
            cit += ' ' + prop['booktitle']
            if 'pages' in prop:
                cit += ', ' + '-'.join(prop['pages'])
        else:
            print "WARNING: Cannot handle bibtex item type:", cat

        cit += '.'

        # beautify citation with linebreaks and proper indentation
        # damn, no. list label has to be a single line... :(
        #rst += format_property(cit, 0)
        rst += cit

        # place optional paper summary
        if 'pymvpa-summary' in prop:
            rst += '\n  *' + format_property(prop['pymvpa-summary'], 2) + '*\n'

        # make keywords visible
        if 'pymvpa-keywords' in prop:
            rst += '\n  Keywords: ' \
                   + ', '.join([':keyword:`' + kw.strip() + '`' 
                                for kw in prop['pymvpa-keywords'].split(',')]) \
                   + '\n'

        # place DOI link
        if 'doi' in prop:
            rst += '\n  DOI: '
            if not prop['doi'].startswith('http://dx.doi.org/'):
                rst += 'http://dx.doi.org/'
            rst += prop['doi']
            rst += '\n'
        # use URL (even if DOI is available -- might lead to a copy outside of the paywall)
        if 'url' in prop:
            rst += '\n  URL: ' + prop['url'] + '\n'

        rst += '\n\n'

    # end list with blank line
    rst += '\n\n'

    return rst.encode('utf-8')


##REF: Name was automagically refactored
def smooth_rst(s):
    """Replace problematic stuff with less problematic stuff."""
    s = re.sub("``", '"', s)
    # assuming that empty strings to not occur in a bib file
    s = re.sub("''", '"', s)

    return s


# do it
bib = BibTeX('doc/misc/references.bib')

refpage = open('doc/source/references.rst', 'w')
refpage.write(bib2rst_references(bib))
refpage.close()

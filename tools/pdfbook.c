/*
 * pdfbook.c    Rearrange pages in a PDF file into signatures.
 *
 * Authors:     Tigran Aivazian <tigran at aivazian.fsnet.co.uk>
 *              Jaap Eldering <eldering at a-eskwadraat.nl>
 *              Roman Buchert <roman.buchert at arcor.de>
 *              Pierre Francois <pf at romanliturgy.org>
 *
 * Based on the algorithm from psutils/psbook.c, which was
 * written by Angus J. C. Duggan 1991-1995.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#define TMP_INFILE_BASE  "input"
#define TMP_OUTFILE_BASE "output"
#define DEF_PAPERSIZE    "a4"

char *program;

static void usage(void)
{
	fprintf(stderr,
	        "Usage: %s [OPTION]...                  infile outfile\n"
	        "  or:  %s [-q] [-p <size>] -o <string> infile outfile\n"
	        "Rearrange pages for printing as booklet.\n\n"
	        "Options:\n"
	        "  -q              suppress verbose output\n"
	        "  -d              debug mode: do not cleanup temporary files\n"
	        "  -2              place 2 pages on 1 page of output\n"
	        "  -s <signature>  group pages together in groups of size <signature>\n"
	        "                  <signature> must be positive and divisible by 4\n"
	        "  -r              reduce the last book to minimum required number of pages\n"
	        "  -p <paper>      set the paper size to: a3, a4, a5, b3, b4, b5, letter, legal\n"
	        "                  or executive (default is determined from infile, %s if unknown)\n"
	        "  -o <string>     pass <string> directly to LaTeX pdfpages `includepdf' command\n"
	        "                  see the pdfpages package documentation for possible options\n",
	        program, program, DEF_PAPERSIZE);
	fflush(stderr);
	exit(1);
}

char *alloc_and_copy (char *str) {
	char *new_str;

	if ( (new_str = strdup(str)) == NULL ) {
		fprintf(stderr, "%s: error allocating memory\n", program);
		exit(1);
	}
	return (new_str);
}

int check_papersize (char *p) {
	if ((strlen(p) == 2) &&
	    ((*p == 'a') || (*p == 'b')) && ((p[1] >= '3') && (p[1] <= '5'))) {
		return (1); /* a3, a4, a5, b3, b4 or b5 */
	}
	if ((strcmp(p, "letter") == 0) ||
	    (strcmp(p, "legal") == 0) ||
	    (strcmp(p, "executive") == 0)) {
		return (1); /* letter, legal or executive */
	} 
	return (0); /* invalid paper size */
}

char *make_tempdir()
{
	static char dirtemplate[L_tmpnam+10];
	char *dirname;
	
	if ( (dirname = tempnam(NULL, "pdfbk")) == NULL ) {
		fprintf(stderr, "%s: error generating temporary directory\n", program);
		exit(1);
	}
	strcpy(dirtemplate, dirname);
	strcat(dirtemplate, "XXXXXX");

	if ( (dirname = mkdtemp(dirtemplate)) == NULL ) {
		fprintf(stderr, "%s: error generating temporary directory\n", program);
		exit(1);
	}
	
	return dirname;
}

char *allocstr(char *format, ...)
{
	va_list ap;
	char *str;
	char tmp[2];
	int len, n;
	
	va_start(ap,format);
	len = vsnprintf(tmp,1,format,ap);
	va_end(ap);
	
	if ( (str = (char *) malloc(len+1))==NULL ) return NULL;
	
	va_start(ap,format);
	n = vsnprintf(str,len+1,format,ap);
	va_end(ap);
	
	if ( n==-1 || n>len ) {
		fprintf(stderr, "%s: error allocating memory\n", program);
		exit(1);
	}
	
	return str;
}

int main(int argc, char *argv[])
{
	char *infile  = NULL;
	char *outfile = NULL;
	char *tmpdir;
	char *tmptexfile;
	char *tmpinfile;
	char *tmpoutfile;

	FILE *fp, *fout;
	
	int *actualpg;
	
	static char cmdline[1024];
	
	int quiet = 0;
	int debug = 0;
	int nup = 0;
	int pdfcustom = 0;
	int signature = 0;
	int reducelastbook = 0;
	int npages = 0;
	int maxpage;
	int completebooks = 0;
	int restpages = 0;
	int restsignature = 0;
	char *pdfcustom_str = NULL;
	char *papersize_str = NULL;
	char origpapersize[32];
	
	int i, c;
	
	program = argv[0];

	tmpdir = make_tempdir();

	while ( (c = getopt(argc, argv, "2qds:o:p:r")) != -1 ) {
		switch (c) {
		case 's':
			signature = atoi(optarg);
			if (signature < 1 || signature % 4)
				usage();
			break;
		
		case 'r':
			reducelastbook = 1;
			break;
			
		case '2':
			nup = 1;
			break;
			
		case 'o':
			pdfcustom = 1;
			pdfcustom_str = alloc_and_copy(optarg);
			break;

		case 'p':
			papersize_str = alloc_and_copy(optarg);
			if (check_papersize(papersize_str) == 0) {
				fprintf(stderr, "%s: bad paper size `%s'\n", program, papersize_str);
				usage ();
			}
			break;

		case 'q':
			quiet = 1;
			break;

		case 'd':
			debug = 1;
			break;

		default:
			usage();
		}
	}

	if ( optind<argc ) {
		infile = argv[optind++];
	} else {
		fprintf(stderr, "%s: input file must be specified\n", program);
		usage();
	}
			
	if ( optind<argc ) {
		outfile = argv[optind++];
	} else {
		fprintf(stderr, "%s: output file must be specified\n", program);
		usage();
	}

	if ( pdfcustom && (nup || signature || reducelastbook) ) {
		fprintf(stderr, "%s: option -o cannot be combined with other options\n", program);
		usage();
	}
	
	sprintf(cmdline, "pdfinfo %s | sed -ne 's/Pages: *\\([0-9]*\\)/\\1/p;s/Page size:.*(\\([^)]*\\).*/\\1/p'", infile);
	fp = popen(cmdline, "r");
	
	i = fscanf(fp, "%d\n%s", &npages, origpapersize);
	if (i < 1) {
		fprintf(stderr, "%s: error reading npages from \"%s\"\n", program, infile);
		exit(1);
	}
	if ( papersize_str==NULL ) {
		if (i < 2) {
			if (!quiet)
				fprintf(stderr, "%s: warning: cannot determine papersize: using %s\n",
				        program, DEF_PAPERSIZE);
			papersize_str = alloc_and_copy(DEF_PAPERSIZE);
		} else {
			papersize_str = alloc_and_copy(origpapersize);
			for(i=0; i<strlen(papersize_str); i++)
				papersize_str[i] = tolower(papersize_str[i]);
		}
	}
	
	if (npages < 1) {
		fprintf(stderr, "%s: invalid number of pages=%d\n", program, npages);
		exit(1);
	}

	tmptexfile = allocstr("%s/%s.tex",tmpdir,TMP_OUTFILE_BASE);
	tmpinfile  = allocstr("%s/%s.pdf",tmpdir,TMP_INFILE_BASE);
	tmpoutfile = allocstr("%s/%s.pdf",tmpdir,TMP_OUTFILE_BASE);
	
	fout = fopen(tmptexfile, "w");
	if (!fout) {
		fprintf(stderr, "%s: error opening \"%s\" for write\n",
				program, tmptexfile);
		exit(1);
	}

	if (!signature)
		signature = maxpage = npages + (4 - npages%4)%4;
	else {
		if (!reducelastbook)
			maxpage = npages + (signature - npages%signature)%signature;
		else {
			completebooks = npages / signature;
			restpages = npages - (completebooks * signature);
			restsignature = restpages + (4 - restpages%4)%4;
			maxpage = completebooks * signature + (restsignature);
		}
	}
	

	if ( (actualpg = (int *) calloc(maxpage,sizeof(int))) == NULL ) {
		fprintf(stderr, "%s: error allocating memory\n", program);
		exit(1);
	}
	
	for (i=0; i<maxpage; i++) {
		int actual = i - i%signature;
		
		switch (i%4) {
		case 0:
		case 3:
			actual += signature - 1 - (i%signature)/2;
			break;
		case 1:
		case 2:
			actual += (i%signature)/2;
			break;
		}
		if (actual < npages)
			actualpg[i] = actual + 1;
	}

	fprintf(fout,
	        "\\documentclass{book}\n"
	        "\\usepackage[%spaper]{geometry}\n"
	        "\\usepackage{pdfpages}\n"
	        "\\begin{document}\n", papersize_str);
	
	if (pdfcustom) {
		fprintf(fout, "\\includepdf[%s]{%s.pdf}\n", pdfcustom_str, TMP_INFILE_BASE);
	} else {
		if ( nup ) {
			if (!reducelastbook)
				fprintf(fout,"\\includepdf[pages=-, signature=%d, landscape]{%s.pdf}\n",
				        signature, TMP_INFILE_BASE);
			else {
				if (completebooks) {
					fprintf(fout,"\\includepdf[pages=-%d, signature=%d, landscape]{%s.pdf}\n",
					        (signature * completebooks), signature, TMP_INFILE_BASE);
				}
				fprintf(fout,"\\includepdf[pages=%d-, signature=%d, landscape]{%s.pdf}\n",
				        (signature * completebooks)+1, restsignature, TMP_INFILE_BASE);
			}
		} else {
			fprintf(fout,"\\includepdf[pages={");
			
			for (i=0; i<maxpage; i++) {
				if (actualpg[i]) {
					fprintf(fout, "%d", actualpg[i]);
				} else {
					fprintf(fout, "{}");
				}
				fprintf(fout, "%s", i<maxpage-1 ? "," : "");
			}
			fprintf(fout,"}]{%s.pdf}\n", TMP_INFILE_BASE);
		}
	}
	
	fprintf(fout,"\\end{document}\n");
	
	fclose(fout);
	
	if (!quiet)
		printf("%s: Generating output file now, please wait...\n", program);
	
	sprintf(cmdline, "cp %s %s", infile, tmpinfile);
	if (system(cmdline)) {
		fprintf(stderr, "%s: Failed to copy \"%s\" file to \"%s\"\n",
		        program, tmpoutfile, tmpinfile);
		exit(1);
	}
	
	sprintf(cmdline, "cd %s && pdflatex %s > /dev/null 2>&1 < /dev/null",
			tmpdir, tmptexfile);
	if (system(cmdline)) {
		fprintf(stderr, "%s: Failed to generate output, see \"%s/%s.log\" for details\n",
		        program, tmpdir, TMP_OUTFILE_BASE);
		exit(1);
	}
	
	sprintf(cmdline, "cp %s %s", tmpoutfile, outfile);
	if (system(cmdline)) {
		fprintf(stderr, "%s: Failed to write \"%s\" file\n",
		        program, outfile);
		exit(1);
	}
	
	if (!debug) {
		sprintf(cmdline, "rm -rf %s", tmpdir);
		system(cmdline);
	}
	
	return 0;
}
